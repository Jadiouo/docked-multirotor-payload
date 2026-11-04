"""評估 rl_train 的 checkpoint：和 g1_gate 同一套評估任務與分數定義（ATT_STEP、POS_STEP、T2，每格 100 個 episode）。

與 scripts/g1_gate.py 的對應（常數直接 import，指標逐行對照 run_cell）：
  - 同樣的環境 seed（SEED[task]）、同樣的 env.reset()，所以初始條件與 B1 完全相同。
    （policy 只用 step 回傳的 obs，不額外呼叫 env.observe，所以不會多吃環境的亂數。）
  - 成功、位置／姿態 RMS（失敗後用 3 m／100° 補）、收斂、分數 = success × max(0, 1 − e_rms/e_ref)。
  - 姿態誤差用 B1 控制器的外迴路定義（R 與 R_d 的夾角），和古典法同一把尺。
  - 同一構型的 B1 直接呼叫 g1_gate.run_cell 重算（CPU）。B1-QP 的 scipy 求解很慢，預設讀 results/p2/g1_raw.json
    （增益與 episode 數必須一致，且該格存在才用）；加 --recompute-qp 則重算。
RL 用 deterministic 動作（Gaussian 的 mean）。輸出 JSON：每格的成功率、分數、RMS，以及對 B1 的比值。
--payload COND（E1，固定點）：用 g2 完全相同的 payload 條件（payload10 / payload15 / all15，來自 scripts/g2_mismatch.CONDS 與 perturb_of）
與同樣的環境 seed 評估；對照 B1、B1-QP、B3 系列（B3 INDI、B3+adapt）較強者、B1-QP-informed（特權 oracle），
--holdout（E2，留出集）：質量 {0.10, 0.15, 0.20} x 掛點每個 episode 在所有模組中隨機（env seed = g1 的 SEED[task]，
與訓練 seed*1000+i 分開、固定寫死），每個質量 100 episode；古典法 B1、B1-QP、B3 系列用同一個 env（同 seed 成對）重算並快取，
B1-QP-informed 需要逐 episode 的 host，目前不支援（只能 E1）。
--payload E3：偏離中心固定點（payload 15%、掛 argmax(x) 模組、offset = p3d_factorial.OFFSET = (+0.0333, −0.0333, −0.0215) m），
  對照 B1-QP、B1-QP-yawlim、B1-QP-estim-yawlim（snap=False）、B1-QP-informed-offset-yawlim（特權：知道偏移），快取 key 以 "E3|" 開頭。
--event：RQ-D 中途 payload 事件評估（attach／drop x 中心／偏離中心，payload 15%、掛點 random、事件 U(1.0, 3.5) s；任務 T2、POS_STEP、HOVER），
  指標同 scripts/rqd_classical.py（分數、成功率、事件後 1 s 位置誤差峰值、恢復時間），並與 results/p2/rqd_classical_raw.json 的古典方法逐 episode 成對比較
  （該檔由 scripts.rqd_classical --stage report 產生；沒有就只報 RL）。
輸出各自的分數比值與成對差（同 seed 同初始條件，逐 episode 相減，bootstrap 95% CI）。
古典法優先讀 results/p2/g2_raw.json、g2_informed_raw.json（條件前綴在 key 裡，增益與 episode 數一致才用），
沒有的格重算並存到 results/p3/rl_eval_cache.json（key 一樣含條件，不同條件不會混用）。
用法（專案根目錄，CPU 即可）：
  CUDA_VISIBLE_DEVICES= .venv-gpu/bin/python -m scripts.rl_eval results/p3/<run>/final.pt --configs single [--out x.json]
"""
import argparse
import contextlib
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import torch

from quadsim.env import ModularEnv
from quadsim.controllers import make_controller, goal_of
from quadsim.policy import ModularActorCritic
from quadsim.rl_env import PackedEnv, RandomPayloadEnv, ResidualEnv, make_estim_yawlim_controller
from scripts.g1_gate import (RES, TASKS, SEED, E_REF, CONV_THR, CONV_FROM, FILL_POS, FILL_ATT,
                             ALL_CONFIGS, load_gains, run_cell)


def load_policy(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    agent = ModularActorCritic(**ck["hp"]).to(device)
    agent.load_state_dict(ck["model"])
    a = ck.get("args") or {}
    if not a and (Path(path).parent / "args.json").exists():
        a = json.loads((Path(path).parent / "args.json").read_text())
    # 殘差 RL：從訓練 args 讀出 scale（None = 一般 policy）。評估時 env 會包 ResidualEnv，古典對照組不受影響
    agent.residual_scale = float(a.get("residual_scale", 0.2)) if a.get("residual") else None
    return agent.eval(), ck


def wrap_residual(agent, env):
    """agent 是殘差 policy 時把 env 包成 ResidualEnv（u_base = estim-yawlim，snap=False）；否則原樣回傳。"""
    sc = getattr(agent, "residual_scale", None)
    return env if sc is None else ResidualEnv(env, make_estim_yawlim_controller(), sc)


HOLDOUT_FRACS = (0.10, 0.15, 0.20)


def run_cell_rl(agent, cells, task, n_eps, gains, device="cpu", perturb=None, holdout_frac=None, seed_offset=0):
    """和 g1_gate.run_cell 同一套指標，只是動作由 policy 給。回傳同格式的 dict（每個 episode 一個值）。"""
    if holdout_frac is not None:
        env = RandomPayloadEnv(cells, n_eps, holdout_frac, frac_choices=(holdout_frac,), task=task, seed=SEED[task] + seed_offset, device=device)
    else:
        env = ModularEnv(cells, n_eps, task=task, seed=SEED[task] + seed_offset, device=device, perturb=perturb)
    pe = PackedEnv(wrap_residual(agent, env), agent.n_max, agent.hist)
    ref = make_controller("B1", gains)                    # 只用它的 attitude_error（外迴路的 R_d）
    ref._env = None
    ref.act(env.state(), goal_of(env.state()), env)       # 讓它用 env 的標稱形狀完成 setup
    obs, _ = pe.reset()
    T = env.cfg.episode_steps
    pos_err, att_err, jload = (np.zeros((T, n_eps)) for _ in range(3))
    term = np.zeros((T, n_eps), bool)
    for s in range(T):
        with torch.no_grad():
            action, _ = agent.act(obs, deterministic=True)
        obs, _, _, te, _, info = pe.step(action)
        fs = info["final_state"]                          # reset 之前的狀態
        pos_err[s] = (fs["goal_pos"] - fs["pos"]).norm(dim=-1).cpu().numpy()
        att_err[s] = ref.attitude_error(fs, goal_of(fs)).cpu().numpy()
        jload[s] = info["joint_load_ratio"].cpu().numpy()
        term[s] = te.cpu().numpy()
    # ↓ 以下與 g1_gate.run_cell 逐行相同
    died = np.maximum.accumulate(term, axis=0)
    valid = ~np.vstack([np.zeros((1, n_eps), bool), died[:-1]])
    success = ~died[-1]
    pos_f = np.where(valid, pos_err, FILL_POS)
    att_f = np.where(valid, att_err, FILL_ATT)
    pos_rms, att_rms = np.sqrt((pos_f ** 2).mean(0)), np.sqrt((att_f ** 2).mean(0))
    jmax = np.where(valid, jload, 0).max(0)
    if task == "T2":
        ends = [i * 150 - 2 for i in (1, 2, 3)]
        conv = np.mean([(pos_f[e] < CONV_THR[task]) & valid[e] for e in ends], 0)
    else:
        e = att_f if task == "ATT_STEP" else pos_f
        conv = ((e[CONV_FROM:] < CONV_THR[task]).all(0) & success).astype(float)
    err = att_rms if task == "ATT_STEP" else pos_rms
    score = success * np.clip(1 - err / E_REF[task], 0, None)
    return dict(success=success.astype(int).tolist(), att_rms=att_rms.tolist(), pos_rms=pos_rms.tolist(),
                conv=conv.tolist(), jmax=jmax.tolist(), score=score.tolist())


def summarize(cell, base=None, base_qp=None):
    m = lambda k: float(np.mean(cell[k]))
    out = dict(success=m("success"), score=m("score"), att_rms=m("att_rms"), pos_rms=m("pos_rms"), conv=m("conv"))
    if base:
        out["ratio_vs_B1"] = out["score"] / base["score"] if base["score"] > 0.05 else math.nan
    if base_qp:     # 正規化分母用兩個 oracle 中較強的那個（PROGRESS：max(B1, B1-QP)）
        best = max(base["score"], base_qp["score"])
        out["ratio_vs_best_oracle"] = out["score"] / best if best > 0.05 else math.nan
    return out


def qp_cell(cfg, task, n_eps, gains, recompute, seed_offset=0):
    """B1-QP 的格子：優先讀 g1_raw.json 的快取（增益與 episode 數一致才用），否則重算。
    g1_raw.json 是舊 seed 跑的，seed_offset != 0 時一律重算。"""
    raw_path = RES / "g1_raw.json"
    if seed_offset == 0 and not recompute and raw_path.exists():
        raw = json.loads(raw_path.read_text())
        if raw["_meta"] == dict(gains=gains.to_dict(), n_eps=n_eps) and f"{cfg}|B1-QP|{task}" in raw:
            return raw[f"{cfg}|B1-QP|{task}"]
    with seed_shift(seed_offset):
        return run_cell(ALL_CONFIGS[cfg], "B1-QP", task, n_eps, gains)


# ---------------------------------------------------------------------------- g2 payload 條件
INDI_SERIES = ["B3 INDI", "B3+adapt"]
CACHE = RES.parent / "p3" / "rl_eval_cache.json"


@contextlib.contextmanager
def seed_shift(off):
    """古典端的 run_cell 在 g1_gate 內以 SEED[task] 取 env seed；暫時把 g1_gate.SEED 全部加上 off（off=0 不動）。"""
    import scripts.g1_gate as g1
    if not off:
        yield
        return
    orig = g1.SEED
    g1.SEED = {t: v + off for t, v in orig.items()}
    try:
        yield
    finally:
        g1.SEED = orig


def cache_path(off=0):
    """off=0 用原本的 CACHE；其他 offset 各用獨立檔（避免互相覆蓋）。"""
    return CACHE if off == 0 else CACHE.with_name(f"{CACHE.stem}_so{off}{CACHE.suffix}")


def ckey(k, off=0):
    """快取 key：off=0 與舊版相同，其他加 so<off>| 前綴。"""
    return k if off == 0 else f"so{off}|{k}"


def cache_load(meta, off=0):
    """讀快取；_meta（含 seed_offset）不一致就視為空。舊檔 _meta 沒有 seed_offset 者視為 0。"""
    meta = dict(meta, seed_offset=off)
    path = cache_path(off)
    cache = json.loads(path.read_text()) if path.exists() else {}
    cm = dict(cache.get("_meta") or {})
    if cm: cm.setdefault("seed_offset", 0)
    if cm != meta: cache = {}
    cache["_meta"] = meta
    return cache


def cache_save(cache, off=0):
    path = cache_path(off)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache))


def _read_json(path):
    return json.loads(Path(path).read_text())


def _g2_cell(cond, cfg, method, task, n_eps, gains, name=None, ckw=None, perturb=None, seed_offset=0):
    """g2 條件下某方法的格子（dict，含逐 episode 的 score）。順序：g2 原始檔（僅 seed_offset=0）→ 本檔快取 → 重算並寫入快取。"""
    from scripts.g2_mismatch import key
    meta = dict(gains=gains.to_dict(), n_eps=n_eps)
    k = key(cond, cfg, method, task)
    if seed_offset == 0:        # g2_raw／g2_informed_raw 是舊 seed 跑的，只有 offset=0 才能用
        for fn in ("g2_raw.json", "g2_informed_raw.json"):
            path = RES / fn
            if path.exists():
                raw = _read_json(path)
                if raw.get("_meta") == meta and k in raw:
                    return raw[k]
    cache = cache_load(meta, seed_offset)
    ck = ckey(k, seed_offset)
    if ck not in cache:
        with seed_shift(seed_offset):
            cache[ck] = run_cell(ALL_CONFIGS[cfg], name or method, task, n_eps, gains, perturb=perturb, ckw=ckw)
        cache_save(cache, seed_offset)
    return cache[ck]


def estim_cell(cfg, task, n_eps, gains, perturb=None, holdout_frac=None, seed_offset=0, tag="E1"):
    """estim-yawlim（B1-QP-estim-yawlim，snap=False；參數 = p3d_factorial.est_params()／yaw_params()，
    來源 results/p2/g2_estim_tune.json、g1_coord_tune.json，與 rqd_classical.method_spec 相同）。
    holdout_frac 非 None 時用 RandomPayloadEnv（E2）。結果進快取（key 含 seed_offset）。"""
    import scripts.g1_gate as g1
    import quadsim.controllers as qc
    from scripts.p3d_factorial import est_params, yaw_params
    meta = dict(gains=gains.to_dict(), n_eps=n_eps)
    cache = cache_load(meta, seed_offset)
    ck = ckey("|".join((tag, cfg, "B1-QP-estim-yawlim", task, "" if holdout_frac is None else f"{holdout_frac:.2f}")), seed_offset)
    if ck in cache: return cache[ck]
    y = yaw_params()
    g = dataclasses.replace(gains, yaw_ratio=y["yaw_ratio"])
    ckw = {**dict(rate_c=y["rate_c"]), **est_params(), "snap": False}
    orig_mk, orig_env = g1.make_controller, g1.ModularEnv
    g1.make_controller = qc.make_controller_p3d_off
    if holdout_frac is not None:
        fr = holdout_frac
        g1.ModularEnv = lambda c, n, **kw: RandomPayloadEnv(c, n, fr, frac_choices=(fr,), **{k: v for k, v in kw.items() if k != "perturb"})
    try:
        with seed_shift(seed_offset):
            r = run_cell(ALL_CONFIGS[cfg], "B1-QP-estim-yawlim", task, n_eps, g, perturb=perturb, ckw=ckw)
    finally:
        g1.make_controller, g1.ModularEnv = orig_mk, orig_env
    cache[ck] = {k: r[k] for k in ("score", "success", "pos_rms", "att_rms")}
    cache_save(cache, seed_offset)
    return cache[ck]


def _boot_diff(a, b, nb=1000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, len(d), (nb, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def compare_g2(rl_cell, cond, cfg, task, n_eps, gains, seed_offset=0):
    """RL 對 g2 基準的比值與成對差。回傳 dict：基準名 → {score, success, ratio, diff, ci}。"""
    from scripts.g2_mismatch import perturb_of
    pt = perturb_of(cond, ALL_CONFIGS[cfg])
    ck = dict(payload_frac=pt.payload_frac, payload_host=pt.payload_host)
    so = seed_offset
    cells = {"B1": _g2_cell(cond, cfg, "B1", task, n_eps, gains, perturb=pt, seed_offset=so),
             "B1-QP": _g2_cell(cond, cfg, "B1-QP", task, n_eps, gains, perturb=pt, seed_offset=so)}
    indi = {m: _g2_cell(cond, cfg, m, task, n_eps, gains, perturb=pt, seed_offset=so) for m in INDI_SERIES}
    best = max(INDI_SERIES, key=lambda m: np.mean(indi[m]["score"]))
    cells[f"{best}（B3 系列最強）"] = indi[best]
    cells["B1-QP-informed"] = _g2_cell(cond, cfg, "B1-QP-informed", task, n_eps, gains, perturb=pt, ckw=ck, seed_offset=so)
    cells["B1-QP-estim-yawlim（snap=False）"] = estim_cell(cfg, task, n_eps, gains, perturb=pt, seed_offset=so, tag=f"E1-{cond}")
    rl = np.asarray(rl_cell["score"], float)
    out = {}
    for name, c in cells.items():
        sc = np.asarray(c["score"], float)
        d, lo, hi = _boot_diff(rl, sc)
        out[name] = dict(score=float(sc.mean()), success=float(np.mean(c["success"])),
                         ratio=float(rl.mean() / sc.mean()) if sc.mean() > 0.05 else math.nan, diff=d, ci=[lo, hi])
    return out


def compare_holdout(rl_cell, frac, cfg, task, n_eps, gains, seed_offset=0):
    """E2：古典法在同樣的 RandomPayloadEnv（同 seed → 同 frac／host 序列）上重算，快取 key 含 holdout 條件與 seed_offset。"""
    import scripts.g1_gate as g1
    cond = f"holdout{frac:.2f}"
    meta = dict(gains=gains.to_dict(), n_eps=n_eps)
    cache = cache_load(meta, seed_offset)
    orig = g1.ModularEnv
    g1.ModularEnv = lambda cells, n, **kw: RandomPayloadEnv(cells, n, frac, frac_choices=(frac,), **{k: v for k, v in kw.items() if k != "perturb"})
    try:
        cells = {}
        with seed_shift(seed_offset):
            for m in ["B1", "B1-QP", *INDI_SERIES]:
                k = ckey("|".join((cond, cfg, m, task)), seed_offset)
                if k not in cache:
                    cache[k] = run_cell(ALL_CONFIGS[cfg], m, task, n_eps, gains)
                    cache_save(cache, seed_offset)
                cells[m] = cache[k]
    finally:
        g1.ModularEnv = orig
    best = max(INDI_SERIES, key=lambda m: np.mean(cells[m]["score"]))
    out, rl = {}, np.asarray(rl_cell["score"], float)
    es = estim_cell(cfg, task, n_eps, gains, holdout_frac=frac, seed_offset=seed_offset, tag="E2")
    for name, c in [("B1", cells["B1"]), ("B1-QP", cells["B1-QP"]), (f"{best}（B3 系列最強）", cells[best]),
                    ("B1-QP-estim-yawlim（snap=False）", es)]:
        sc = np.asarray(c["score"], float)
        d, lo, hi = _boot_diff(rl, sc)
        out[name] = dict(score=float(sc.mean()), success=float(np.mean(c["success"])),
                         ratio=float(rl.mean() / sc.mean()) if sc.mean() > 0.05 else math.nan, diff=d, ci=[lo, hi])
    return out


E3_METHODS = [("B1-QP", dict(name="B1-QP")), ("B1-QP-yawlim", dict(name="B1-QP-yawlim")),
              ("B1-QP-estim-yawlim（snap=False）", dict(name="B1-QP-estim-yawlim", nosnap=True)),
              ("B1-QP-informed-offset-yawlim（特權）", dict(name="B1-QP-informed-offset-yawlim", informed=True))]


def e3_perturb(cells):
    from scripts.p3d_factorial import cond_perturb
    return cond_perturb("offcenter", cells)


def compare_e3(rl_cell, cfg, task, n_eps, gains, seed_offset=0):
    """E3：古典法用 p3d_factorial.run_one（同 seed、同偏移）重算，快取到 rl_eval_cache.json（key 以 E3| 開頭）。"""
    from scripts.p3d_factorial import run_one
    meta = dict(gains=gains.to_dict(), n_eps=n_eps)
    cache = cache_load(meta, seed_offset)
    out, rl = {}, np.asarray(rl_cell["score"], float)
    for label, kw in E3_METHODS:
        k = ckey("|".join(("E3", cfg, label, task)), seed_offset)
        if k not in cache:
            with seed_shift(seed_offset):
                cache[k] = run_one(cfg, "offcenter", kw["name"], task, n_eps, gains, nosnap=kw.get("nosnap", False), informed=kw.get("informed", False))
            cache_save(cache, seed_offset)
        sc = np.asarray(cache[k]["score"], float)
        d, lo, hi = _boot_diff(rl, sc)
        out[label] = dict(score=float(sc.mean()), success=float(np.mean(cache[k]["success"])),
                          ratio=float(rl.mean() / sc.mean()) if sc.mean() > 0.05 else math.nan, diff=d, ci=[lo, hi])
    return out


def run_rl_event(agent, cells, task, n_eps, typ, off, device="cpu", seed_offset=0):
    """RQ-D：RL 在中途 payload 事件下，指標同 scripts/rqd_classical.run_ev（同 env seed、同事件時刻與掛點）。"""
    from scripts.rqd_classical import event_of, metrics_from_traces, SEED_EV
    from quadsim.env import EnvConfig, Perturb
    env = ModularEnv(cells, n_eps, task="POS_STEP" if task == "HOVER" else task, seed=SEED_EV[task] + seed_offset, device=device,
                     cfg=EnvConfig(pos_step=0.0) if task == "HOVER" else None, perturb=Perturb(payload_event=event_of(typ, off)))
    pe = PackedEnv(wrap_residual(agent, env), agent.n_max, agent.hist)
    obs, _ = pe.reset()
    t_ev = env.t_event.clone().cpu().numpy()
    T = env.cfg.episode_steps
    pos_err, term = np.zeros((T, n_eps)), np.zeros((T, n_eps), bool)
    for s in range(T):
        with torch.no_grad():
            action, _ = agent.act(obs, deterministic=True)
        obs, _, _, te, _, info = pe.step(action)
        fs = info["final_state"]
        pos_err[s] = (fs["goal_pos"] - fs["pos"]).norm(dim=-1).cpu().numpy(); term[s] = te.cpu().numpy()
    return metrics_from_traces(pos_err, term, task, t_ev)


def compare_event(rl, cfg, task, typ, off, n_eps=None, gains=None, seed_offset=0):
    """與古典方法逐 episode 成對（分數差 bootstrap CI）。seed_offset=0：讀 results/p2/rqd_classical_raw.json（舊 seed）；
    seed_offset!=0：用 rqd_classical.run_ev 以 SEED_EV+offset 重算並快取（key 含 seed_offset）。"""
    methods = ("B1-QP-yawlim", "estim-yawlim", "estim-reset-yawlim", "informed-event-yawlim")
    cells = {}
    if seed_offset == 0:
        path = RES / "rqd_classical_raw.json"
        if not path.exists(): return {}
        raw = _read_json(path)["cells"]
        for m in methods:
            c = raw.get(f"{cfg}|{task}|{typ}-{off}|{m}")
            if c is not None: cells[m] = c
    else:
        from scripts.rqd_classical import run_ev, SEED_EV
        cache = cache_load(dict(gains=gains.to_dict(), n_eps=n_eps), seed_offset)
        for m in methods:
            k = ckey("|".join(("EV", cfg, task, f"{typ}-{off}", m)), seed_offset)
            if k not in cache:
                cache[k] = run_ev(ALL_CONFIGS[cfg], m, task, n_eps, gains, SEED_EV[task] + seed_offset, typ, off)
                cache_save(cache, seed_offset)
            cells[m] = cache[k]
    out = {}
    for m, c in cells.items():
        sc = np.asarray(c["score"], float)
        d, lo, hi = _boot_diff(rl["score"], sc)
        out[m] = dict(score=float(sc.mean()), success=float(np.mean(c["success"])), peak=float(np.nanmean(c["peak"])),
                      rec_median_s=float(np.median(np.asarray(c["rec_steps"]) / 100.0)), diff=d, ci=[lo, hi])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--configs", nargs="+", default=["single"])
    ap.add_argument("--tasks", nargs="+", default=None, help="預設 ATT_STEP POS_STEP T2；--event 預設 T2 POS_STEP HOVER")
    ap.add_argument("--n-eps", type=int, default=100)
    ap.add_argument("--recompute-qp", action="store_true")
    ap.add_argument("--payload", default=None, choices=["payload10", "payload15", "all15", "E3"],
                    help="用 g2 的 payload 條件評估（與 g2 同 seed），並對照 B1-QP、B3 系列最強、B1-QP-informed")
    ap.add_argument("--event", action="store_true", help="RQ-D 中途 payload 事件評估（attach／drop x 中心／偏離中心）")
    ap.add_argument("--holdout", action="store_true", help="E2 留出集：質量 {0.10,0.15,0.20} x 隨機掛點")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="評估 env seed = 原 seed + offset（探索 0、驗證 100、confirmatory 900000）；古典端同步重算，不讀舊 seed 的 g1/g2/rqd 原始檔")
    ap.add_argument("--cache", default=None, help="古典端快取檔（預設 results/p3/rl_eval_cache.json；offset!=0 另加 _so<offset> 後綴）")
    ap.add_argument("--out", default=None, help="輸出 JSON；預設與 checkpoint 同資料夾的 eval_<檔名>.json")
    args = ap.parse_args()
    torch.set_num_threads(2)
    global CACHE
    if args.cache: CACHE = Path(args.cache)
    so = args.seed_offset
    if args.tasks is None: args.tasks = ["T2", "POS_STEP", "HOVER"] if args.event else TASKS
    agent, ck = load_policy(args.ckpt)
    gains = load_gains()
    if args.payload and args.payload != "E3":
        from scripts.g2_mismatch import perturb_of
    result = dict(payload=args.payload, holdout=args.holdout, event=args.event, ckpt=args.ckpt, iter=ck["iter"], train_args=ck["args"], n_eps=args.n_eps, seed_offset=so, cells={})
    for cfg in args.configs:
        for task in args.tasks:
            if args.event:
                for typ in ("attach", "drop"):
                    for off in ("center", "offcenter"):
                        rl = run_rl_event(agent, ALL_CONFIGS[cfg], task, args.n_eps, typ, off, seed_offset=so)
                        cmp = compare_event(rl, cfg, task, typ, off, args.n_eps, gains, so)
                        pk, rc = float(np.nanmean(rl["peak"])), float(np.median(np.asarray(rl["rec_steps"]) / 100.0))
                        result["cells"][f"{cfg}|{task}|{typ}-{off}"] = dict(RL=dict(success=float(np.mean(rl["success"])), score=float(np.mean(rl["score"])),
                                                                                  peak=pk, rec_median_s=rc, recovered=float(np.mean(rl["recovered"]))), vs=cmp)
                        print(f"{cfg:8s} {task:8s} [event {typ}-{off}] RL 成功率 {np.mean(rl['success']):.2f} 分數 {np.mean(rl['score']):.3f} 峰值 {pk:.3f} m 恢復 {rc:.2f} s", flush=True)
                        for n, v in cmp.items():
                            print(f"    vs {n:22s} 分數 {v['score']:.3f} 成功率 {v['success']:.2f} 峰值 {v['peak']:.3f} 成對差 {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]", flush=True)
                continue
            if args.holdout:
                for fr in HOLDOUT_FRACS:
                    rl_raw = run_cell_rl(agent, ALL_CONFIGS[cfg], task, args.n_eps, gains, holdout_frac=fr, seed_offset=so)
                    cmp = compare_holdout(rl_raw, fr, cfg, task, args.n_eps, gains, so)
                    s = summarize(rl_raw)
                    result["cells"][f"{cfg}|{task}|holdout{fr:.2f}"] = dict(RL=s, vs=cmp)
                    print(f"{cfg:8s} {task:8s} [E2 frac={fr:.2f} 隨機掛點] RL 成功率 {s['success']:.2f} 分數 {s['score']:.3f}", flush=True)
                    for n, v in cmp.items():
                        print(f"    vs {n:22s} 分數 {v['score']:.3f} 成功率 {v['success']:.2f} 比值 {v['ratio']:.3f} "
                              f"成對差 {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]", flush=True)
                continue
            if args.payload:
                if args.payload == "E3":
                    pt = e3_perturb(ALL_CONFIGS[cfg])
                    rl_raw = run_cell_rl(agent, ALL_CONFIGS[cfg], task, args.n_eps, gains, perturb=pt, seed_offset=so)
                    cmp = compare_e3(rl_raw, cfg, task, args.n_eps, gains, so)
                else:
                    pt = perturb_of(args.payload, ALL_CONFIGS[cfg])
                    rl_raw = run_cell_rl(agent, ALL_CONFIGS[cfg], task, args.n_eps, gains, perturb=pt, seed_offset=so)
                    cmp = compare_g2(rl_raw, args.payload, cfg, task, args.n_eps, gains, so)
                s = summarize(rl_raw)
                result["cells"][f"{cfg}|{task}"] = dict(RL=s, vs=cmp)
                print(f"{cfg:8s} {task:8s} [{args.payload}] RL 成功率 {s['success']:.2f} 分數 {s['score']:.3f}", flush=True)
                for n, v in cmp.items():
                    print(f"    vs {n:22s} 分數 {v['score']:.3f} 成功率 {v['success']:.2f} 比值 {v['ratio']:.3f} "
                          f"成對差 {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]", flush=True)
                continue
            rl = run_cell_rl(agent, ALL_CONFIGS[cfg], task, args.n_eps, gains, seed_offset=so)
            with seed_shift(so):
                b1 = summarize(run_cell(ALL_CONFIGS[cfg], "B1", task, args.n_eps, gains))
            qp = summarize(qp_cell(cfg, task, args.n_eps, gains, args.recompute_qp, so))
            s = summarize(rl, b1, qp)
            result["cells"][f"{cfg}|{task}"] = dict(RL=s, B1=b1, **{"B1-QP": qp})
            print(f"{cfg:8s} {task:8s} RL 成功率 {s['success']:.2f} 分數 {s['score']:.3f} | "
                  f"B1 {b1['success']:.2f} / {b1['score']:.3f} | B1-QP {qp['success']:.2f} / {qp['score']:.3f} | "
                  f"比值 vs B1 {s['ratio_vs_B1']:.3f}  vs max(B1,QP) {s['ratio_vs_best_oracle']:.3f}", flush=True)
    if args.payload or args.holdout:      # 各任務分數單獨列出（含 T2），另給 T2+POS_STEP 合併
        by = {}
        for ck_, v in result["cells"].items():
            cfg_, task_, *rest = ck_.split("|")
            for name, sc in [("RL", v["RL"]["score"]), *[(n, x["score"]) for n, x in v["vs"].items()]]:
                by.setdefault(f"{cfg_}|{'|'.join(rest)}|{name}", {})[task_] = sc
        for k_, d in by.items():
            if "T2" in d and "POS_STEP" in d: d["T2+POS_STEP"] = (d["T2"] + d["POS_STEP"]) / 2
        result["score_by_task"] = by
        for k_, d in by.items():
            print(f"  [by task] {k_:60s} " + "  ".join(f"{t} {x:.3f}" for t, x in d.items()), flush=True)
    out = Path(args.out) if args.out else Path(args.ckpt).with_name(f"eval_{Path(args.ckpt).stem}.json")
    out.write_text(json.dumps(result, indent=1))
    print("已寫入", out)


if __name__ == "__main__":
    main()
