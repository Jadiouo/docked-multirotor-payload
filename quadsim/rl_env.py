"""RL 用的 env wrapper：pad 到 N_max、維護觀測歷史、打包成 policy 吃的格式（格式見 quadsim/policy.py 檔頭）。

歷史（記憶）放在這裡而不是 policy 裡：policy 是無狀態的，吃「最近 H 步的 obs」。
  obs 的最後 4 維就是上一步的動作 u_prev，所以堆疊 H 步 obs 已經包含了最近 H 步的動作。
  H = 1 就是沒有記憶。episode 一開始，歷史用第一個 obs 填滿（不用 0，避免開頭出現假的跳變）。

補齊：構型的模組數 N < N_max 時，多出來的模組 obs 全為 0、active = 0。
final_obs 的約定同 ppo.py：autoreset 回傳的是新 episode 的 obs，不能拿來 bootstrap，
所以 info["final_obs"] / info["final_critic_obs"] 是 reset 之前那一步的打包結果。
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from .env import ModularEnv, Perturb
from .controllers import Gains, goal_of, make_controller_p3d_off
from .policy import PackLayout, OBS_DIM, PRIV_DIM


def g2_payload_host(cells):
    """g2_mismatch.perturb_of 的掛點規則：掛在 x 最大端的模組（有多個時取第一個），偏移 (0,0,0)。"""
    return int(np.argmax([c[0] for c in cells]))


class RandomPayloadEnv(ModularEnv):
    """每個環境每次 reset 重抽 payload：質量比例 U(0, pmax)（或 frac_choices 時從該集合均勻抽），
    掛點模組在所有模組中均勻抽，偏移 (0,0,0)（同 g2 的做法）。質量比例定義同 Perturb.payload_frac。
    幾何直接呼叫父類別 _sample_geometry（逐環境用該環境的 frac、host），公式與固定 payload 的 env 完全相同。
    actor 的觀測不含 payload；真實 CoM 相對位置 d 會進 PackedEnv 的 critic 特權欄位（見 PackedEnv._priv）。
    self.payload_frac [B]、self.payload_host [B]：目前每個環境的 payload（特權資訊，只給評估／診斷用）。
    module_loads（只影響接頭負載量測，不影響動力學）覆寫成逐環境 host。
    offset_max（預設 0 = 關閉）：payload 掛點偏離模組中心。比例 s = offset_max，各分量的上限取 P3d 偏離中心診斷的偏移大小
      x、y：U(−s·a, s·a)，a = 0.718·0.4·pitch（≈0.0333 m）；z：U(−s·0.718·0.03, 0)（≈ −0.0215 m 往下）。s = 1 時偏移分布以診斷用的固定偏移
      (+0.0333, −0.0333, −0.0215) 的各分量絕對值為界。
    event_prob（預設 0 = 關閉）：每個 episode 以此機率改成中途事件，attach／drop 各半，事件時刻 U(t_range) 秒（預設 1.0–3.5）。
      attach：事件前沒有 payload，事件時刻起有（幾何、連續性見 ModularEnv 的「payload 事件」）；drop：相反。
      沒被選到事件的 episode 維持整段帶 payload（原行為）。self.ev_mode [B]：0 無事件、1 attach、2 drop。"""

    OFFSET_REF = (0.718 * 0.4, 0.718 * 0.4, 0.718 * 0.03)     # 乘上 pitch（x、y）與 1（z, 公尺）見 _offset_bounds

    def __init__(self, cells, n_envs, pmax, frac_choices=None, offset_max=0.0, event_prob=0.0, t_range=(1.0, 3.5), **kw):
        assert 0 < pmax < 1
        self.offset_max, self.event_prob, self.t_range = float(offset_max), float(event_prob), tuple(t_range)
        assert 0 <= self.event_prob <= 1 and self.offset_max >= 0
        self.pmax, self.frac_choices = float(pmax), frac_choices
        self.payload_frac, self.payload_host = torch.zeros(n_envs), torch.zeros(n_envs, dtype=torch.long)
        kw["perturb"] = Perturb(payload_frac=self.pmax, payload_host=0)     # >0 才會走 payload 的分支
        super().__init__(cells, n_envs, **kw)
        self._shared = False
        self._events = self.event_prob > 0
        if self._events:
            self.geo_alt = {k: v.clone() for k, v in self.geo.items()}
            self.reset()                                                 # 父類別的 reset 發生在事件啟用之前，重來一次

    def _offset_bounds(self):
        a = self.offset_max
        return a * 0.718 * 0.4 * self.pitch, a * 0.718 * 0.03

    def _sample_geometry(self, n):
        kw = dict(generator=self.gen, device=self.device)
        if self.frac_choices is None:
            fr = (torch.rand(n, dtype=self.cfg.dtype, **kw) * self.pmax).clamp_min(1e-6)
        else:
            ch = torch.tensor(self.frac_choices, dtype=self.cfg.dtype, device=self.device)
            fr = ch[torch.randint(len(ch), (n,), **kw)]
        hosts = torch.randint(self.N, (n,), **kw)
        offs = [(0., 0., 0.)] * n
        if self.offset_max > 0:                                          # 只有開啟時才抽（關閉時不多吃亂數 → 行為與舊版逐位元相同）
            r = torch.rand(n, 3, dtype=self.cfg.dtype, **kw).cpu().numpy()
            bxy, bz = self._offset_bounds()
            offs = [(float((2 * u[0] - 1) * bxy), float((2 * u[1] - 1) * bxy), float(-u[2] * bz)) for u in r]
        base, parts = self.perturb, []
        try:
            for f, h, o in zip(fr.tolist(), hosts.tolist(), offs):
                self.perturb = dataclasses.replace(base, payload_frac=f, payload_host=h, payload_offset=o)
                parts.append(ModularEnv._sample_geometry(self, 1))
        finally:
            self.perturb = base
        self._last = (fr, hosts)
        return {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}

    def _reset_idx(self, idx):
        new = self._sample_geometry(len(idx))
        if idx.numel() == self.B: self.geo = new
        else:
            for k in new: self.geo[k][idx] = new[k]
        if self._events: self._reset_rand_events(idx, new)
        fr, hosts = self._last
        self.payload_frac = self.payload_frac.to(self.device).clone(); self.payload_host = self.payload_host.to(self.device).clone()
        self.payload_frac[idx] = fr.to(self.payload_frac.dtype); self.payload_host[idx] = hosts
        super()._reset_idx(idx)

    def _reset_rand_events(self, idx, pay):
        """event_prob 的環境改成事件 episode：attach = 事件前無 payload；drop = 事件前帶 payload（事件後無）。
        pay = 本批 payload 幾何。事件模式只用 self.gen（訓練用，不需要跨方法成對）。"""
        n, kw = len(idx), dict(generator=self.gen, device=self.device)
        isev = torch.rand(n, **kw) < self.event_prob
        attach = torch.rand(n, **kw) < 0.5
        lo, hi = self.t_range
        dtc = self.cfg.dt * self.cfg.substeps
        t_ev = torch.floor((lo + (hi - lo) * torch.rand(n, **kw)) / dtc).long()
        non = self._nominal_geo(n)
        pre = {k: torch.where((isev & attach).view(-1, *[1] * (v.dim() - 1)), non[k].to(v.dtype), v) for k, v in pay.items()}
        post = {k: torch.where((isev & ~attach).view(-1, *[1] * (v.dim() - 1)), non[k].to(v.dtype), v) for k, v in pay.items()}
        self._set_event_geo(idx, pre, post)
        self.ev_mode[idx] = torch.where(isev, torch.where(attach, 1, 2), 0)
        self.t_event[idx] = torch.where(isev, t_ev, torch.full_like(t_ev, 10 ** 9))
        self.ev_done[idx] = False

    def _nominal_geo(self, n):
        base = self.perturb
        try:
            self.perturb = dataclasses.replace(base, payload_frac=0.0, payload_offset=(0., 0., 0.))
            return ModularEnv._sample_geometry(self, n)
        finally:
            self.perturb = base

    def module_loads(self):
        base = self.perturb
        self.perturb = dataclasses.replace(base, payload_frac=0.0)           # 先算不含 payload 的部分
        try: out = super().module_loads()
        finally: self.perturb = base
        g = self.geo
        dp = g["dp"]
        Fp = -g["mp"][:, None] * (self._F_b / g["M"][:, None] + torch.cross(self._alpha, dp, dim=-1)
                                  + torch.cross(self.w, torch.cross(self.w, dp, dim=-1), dim=-1))
        oh = torch.nn.functional.one_hot(self.payload_host, self.N).to(Fp.dtype)       # [B,N]
        d_h = (g["d"] * oh[..., None]).sum(1)
        Mp = torch.cross(dp - d_h, Fp, dim=-1)
        out = out.clone()
        out[..., :3] += oh[..., None] * Fp[:, None]
        out[..., 3:] += oh[..., None] * Mp[:, None]
        return out


P2 = Path(__file__).resolve().parent.parent / "results" / "p2"


def make_estim_yawlim_controller():
    """B1-QP-estim-yawlim（snap=False）：參數來源與 scripts/rl_eval.estim_cell／p3d_factorial.run_one 相同
    （classical_gains.json 的增益；g1_coord_tune.json 的 best → yaw_ratio、rate_c；g2_estim_tune.json 的 best → 估計器參數；snap=False）。"""
    gains = Gains(**json.loads((P2 / "classical_gains.json").read_text())["gains"])
    y = json.loads((P2 / "g1_coord_tune.json").read_text())["best"]
    b = json.loads((P2 / "g2_estim_tune.json").read_text())["best"]
    est = {k: b[k] for k in ("warm_c", "warm_m", "lam_m", "eps_m", "lam_c", "eps_c", "snap", "corr")}
    g = dataclasses.replace(gains, yaw_ratio=y["yaw_ratio"])
    return make_controller_p3d_off("B1-QP-estim-yawlim", g, **{"rate_c": y["rate_c"], **est, "snap": False})


class ResidualEnv:
    """殘差 RL 的 env wrapper（探索性，prereg 第 13 節）：u = clip(u_base + scale·δ, −1, 1)。
    u_base = 古典控制器（預設 B1-QP-estim-yawlim）對目前狀態算出的馬達指令；policy 輸出 δ。
    順序與 g1_gate.run_cell 相同：reset 之後與每次 step 之後，用 env 回傳的 post-reset 狀態（info["state"]）算下一步的 u_base，
    所以 δ≡0 時軌跡與直接跑古典法逐位元相同（不額外消耗 env 的亂數）。
    控制器狀態（RLS、yaw 限速）由控制器自己以 state["t"] == 0 逐環境重置：auto-reset 的環境回傳的 post-reset 狀態 t = 0，
    所以下一次 act 時該環境的估計器、yaw_ref 會被重置，其他環境不受影響。
    obs 多接 u_base（每個模組 4 維，共 obs_dim + 4）。info["final_obs"] 同樣多接 4 維：未結束的環境是下一步的 u_base；
    結束（reset）的環境無法在不污染控制器狀態下對 final state 再算一次，改用這一步實際用的 u_base（只影響 truncation 的 value bootstrap，近似）。
    估計器的馬達模型假設自己的指令就是實際推力；殘差會讓實際推力不同，所以把 δ 造成的差（clip 後的 u − u_base）補進控制器的 T_prev
    （δ=0 時補 0，不改變任何數值）。其他屬性（B、N、geo、cfg……）委派給被包的 env。"""

    def __init__(self, env, ctrl=None, scale=0.2):
        self.env, self.scale = env, float(scale)
        self.ctrl = ctrl if ctrl is not None else make_estim_yawlim_controller()
        self.obs_dim = env.obs_dim + 4
        self.u_base = None

    def __getattr__(self, name):          # 只在找不到屬性時才呼叫
        if name == "env": raise AttributeError(name)
        return getattr(self.env, name)

    def _base(self, state):
        return self.ctrl.act(state, goal_of(state), self.env).to(**self.env._tt)

    def reset(self):
        obs = self.env.reset()
        self.u_base = self._base(self.env.state())
        return torch.cat([obs, self.u_base], -1)

    def step(self, delta):
        env, ub = self.env, self.u_base
        u = (ub + self.scale * delta.to(**env._tt)).clamp(-1, 1)
        Tp = getattr(self.ctrl, "T_prev", None)
        if Tp is not None:                                   # 估計器的馬達模型：補上殘差造成的推力差
            Tp += ((u - ub).reshape(env.B, -1).to(Tp.dtype)) * (env.tmax / 2)
            Tp.clamp_(0, env.tmax)
        obs, reward, term, trunc, info = env.step(u)
        nxt = self._base(info["state"])
        done = (term | trunc)[:, None, None]
        info["final_obs"] = torch.cat([info["final_obs"], torch.where(done, ub, nxt)], -1)
        info["u_base"] = ub
        self.u_base = nxt
        return torch.cat([obs, nxt], -1), reward, term, trunc, info


class PackedEnv:
    def __init__(self, env, n_max, hist=1):
        assert env.N <= n_max and env.obs_dim in (OBS_DIM, OBS_DIM + 4)       # +4 = ResidualEnv（多一份 u_base）
        self.env, self.n_max, self.hist = env, n_max, hist
        self.B, self.N, self.device = env.B, env.N, env.device
        self.obs_dim = env.obs_dim
        self.actor_layout = PackLayout(n_max, hist * self.obs_dim)
        self.critic_layout = PackLayout(n_max, self.obs_dim + PRIV_DIM)
        dev = self.device
        self.active = (torch.arange(n_max, device=dev) < env.N).float()                  # [n_max]
        self.adj = torch.zeros(n_max, n_max, 4, device=dev)
        self.adj[:env.N, :env.N] = env.adjacency.float()
        self.hist_buf = None                                                              # [B, n_max, H, obs_dim]

    # ------------------------------------------------------------------ 小工具
    def _pad(self, x):
        """[B, N, f] → [B, n_max, f]，多出來的模組補 0。"""
        out = x.new_zeros(x.shape[0], self.n_max, x.shape[2])
        out[:, :self.N] = x
        return out

    def _priv(self):
        """特權資訊 [B, n_max, 4]：模組中心相對真實 CoM 的位置 / 模組間距（3）、N / N_max（1）。只給 critic。"""
        r = self.env.geo["d"] / self.env.pitch
        n = torch.full_like(r[..., :1], self.N / self.n_max)
        return self._pad(torch.cat([r, n], -1))

    def _pack_critic(self, obs, priv):
        return self.critic_layout.pack(torch.cat([obs, priv], -1), self.adj, self.active)

    def _pack_actor(self, hist):
        return self.actor_layout.pack(hist.reshape(self.B, self.n_max, -1), self.adj, self.active)

    # ------------------------------------------------------------------ 介面
    def reset(self):
        """回傳 (actor_obs [B, n_max, P_a], critic_obs [B, n_max, P_c])。"""
        obs = self._pad(self.env.reset())
        self.hist_buf = obs[:, :, None].expand(-1, -1, self.hist, -1).clone()
        return self._pack_actor(self.hist_buf), self._pack_critic(obs, self._priv())

    def step(self, action):
        """action [B, n_max, 4]（多出來的模組被忽略）。
        回傳 (actor_obs, critic_obs, reward [B], terminated [B], truncated [B], info)；
        info 另含 final_obs / final_critic_obs（reset 之前那一步的打包結果）與 env 原本的 info。"""
        priv_before = self._priv()           # 這一步結束時有 DR 的話，geo 會被 reset 換掉，所以先留一份
        obs, reward, terminated, truncated, info = self.env.step(action[:, :self.N])
        final = self._pad(info["final_obs"])
        hist_final = torch.cat([self.hist_buf[:, :, 1:], final[:, :, None]], 2)           # 含最後一步的歷史
        done = (terminated | truncated)[:, None, None, None]
        new = self._pad(obs)                                                              # done 的環境：新 episode 的第一個 obs
        self.hist_buf = torch.where(done, new[:, :, None].expand(-1, -1, self.hist, -1), hist_final)

        info["final_obs"] = self._pack_actor(hist_final)
        info["final_critic_obs"] = self._pack_critic(final, priv_before)
        return (self._pack_actor(self.hist_buf), self._pack_critic(new, self._priv()),
                reward, terminated, truncated, info)
