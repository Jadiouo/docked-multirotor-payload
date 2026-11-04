"""模組化多旋翼的批次模擬環境（規格見 docs/env_spec.md）。

對接後整架是單一剛體，所以用 torch 對 B 個環境一起做 Newton-Euler 積分，不逐一 step MuJoCo。
MuJoCo 只當參考解（quadsim/mjcf.py，tests/test_env.py 逐步比對）。

座標：世界 z 向上；機體系 x 前、y 左、z 上。四元數 q = (w, x, y, z)，body→world。
轉子順序同 assemble()：模組主序，每模組 4 顆。原點 = 整架真實 CoM（含 payload）。

檔案由上到下：
  1) 四元數工具
  2) EnvConfig（所有可調參數，含 reward 權重）、Perturb（物理誤差）
  3) 幾何：標稱 → 真實（payload、對接偏差、推力係數誤差）
  4) ModularEnv：動力學、接頭內力、觀測、任務與 reset、step
"""
import dataclasses
from dataclasses import dataclass
import math
import numpy as np
import torch

from .assembly import assemble, MOTOR_XY, MOTOR_SPIN
from .module_params import load_module
from .params import G
from analysis.joint_loads import build as build_joint_model, edges as joint_edges, joint_map

TASKS = ("T1", "T2", "ATT_STEP", "POS_STEP")


# ============================================================================
# 1) 四元數 / 旋轉
# ============================================================================
def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_exp(w, dt):
    """機體角速度 w（rad/s）積分 dt 的旋轉四元數 exp(w dt)；|w| 很小時用泰勒展開避免 0/0。"""
    ang = w.norm(dim=-1, keepdim=True) * dt                      # 轉角 θ
    half = ang / 2
    # sin(θ/2)/|w| = dt/2 · sinc(θ/2)；sinc 小角度用 1 − x²/6
    sinc = torch.where(half < 1e-4, 1 - half ** 2 / 6, torch.sin(half) / half.clamp_min(1e-12))
    return torch.cat([torch.cos(half), w * (dt / 2) * sinc], -1)


def quat_to_rot(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def quat_from_euler(roll, pitch, yaw):
    """R = Rz(yaw) Ry(pitch) Rx(roll)。"""
    cr, sr, cp, sp, cy, sy = (f(a / 2) for a in (roll, pitch, yaw) for f in (torch.cos, torch.sin))
    return torch.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], -1)


def rot_xy(tx, ty):
    """小傾斜：先繞 x 轉 tx 再繞 y 轉 ty（機體系），形狀 [..., 3, 3]。"""
    cx, sx, cy, sy = torch.cos(tx), torch.sin(tx), torch.cos(ty), torch.sin(ty)
    o, z = torch.ones_like(tx), torch.zeros_like(tx)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], -1).reshape(*tx.shape, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], -1).reshape(*tx.shape, 3, 3)
    return Rx @ Ry


def matvec(A, x):
    """批次矩陣乘向量 [B,3,3]·[B,3]。CPU 上 einsum/bmm 對這種小矩陣很慢，改用逐元素乘加。"""
    return (A * x[:, None]).sum(-1)


def wrap_pi(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


# ============================================================================
# 2) 設定
# ============================================================================
@dataclass
class EnvConfig:
    # --- 時間 ---
    dt: float = 0.002                  # 物理步長（s）
    substeps: int = 5                  # 每個控制步的物理子步數（100 Hz 控制）
    episode_steps: int = 500           # 5 s
    dtype: torch.dtype = torch.float32
    # --- 感測雜訊（數值為假設）---
    gyro_noise: float = 0.01           # rad/s
    accel_noise: float = 0.05          # m/s²
    blind: bool = False                # True：鄰居旗標全設 0（盲飛）
    # --- 任務 ---
    hover_height: float = 1.5          # 目標 hover 高度（m）；地面在 z=0，所以「回到原點」指 (0, 0, hover_height)
    t1_tilt_max: float = math.radians(60)   # T1：roll、pitch 初值範圍
    t1_pos_max: float = 0.5            # T1：初始位置相對目標的範圍（每軸 ±，m）
    t1_vel_max: float = 1.0            # T1：初速（每軸 ±，m/s）
    t1_rate_max: float = 3.0           # T1：初始角速度（每軸 ±，rad/s）
    t2_period: int = 150               # T2：每 1.5 s 換目標
    t2_pos_max: float = 1.0            # T2：目標位置 ±1 m
    t2_yaw_max: float = math.pi / 2    # T2：目標 yaw ±90°
    att_step: tuple = (0.8, 1.2)       # ATT_STEP：roll 或 pitch 階躍大小（rad）。階躍 = 初始姿態誤差，
                                       #   因為觀測沒有「目標姿態」，策略只看得到相對 hover 姿態的偏差
    pos_step: float = 1.5              # POS_STEP：位置階躍距離（m），方向在水平面隨機
    # --- 終止 ---
    max_pos_err: float = 3.0           # |位置誤差| > 3 m
    max_tilt: float = math.radians(100)
    detach_on_overload: bool = False   # True：接頭負載比 > 1 就終止（info["overload"]）
    # --- reward（初始值，由實作者訂；整隊共用）---
    w_pos: float = 1.0
    w_vel: float = 0.1
    w_rate: float = 0.02
    w_yaw: float = 0.2
    w_action: float = 0.01             # −w·mean(|u − u_prev|²)，對 N×4 個動作取平均（和構型大小無關）
    alive: float = 1.0
    terminal: float = 20.0
    w_joint: float = 0.0               # 接頭負載懲罰 −w·max(負載比)；預設關閉
    props_path: str = None             # module_props.json 路徑；None = 預設


@dataclass
class Perturb:
    """物理誤差。全部預設為 0 = 標稱。對接偏差與推力係數每次 reset 重抽（可當 DR）；payload 固定。"""
    payload_frac: float = 0.0          # payload / (模組總質量 + payload)
    payload_host: int = -1             # 掛在第幾個模組（-1 = 最後一個）
    payload_offset: tuple = (0., 0., 0.)   # 相對該模組中心的位置（m）
    dock_delta: float = 0.0            # 每個模組中心各軸 U(−δ, δ) 的偏移（m）
    dock_tilt: float = 0.0             # 每個模組繞 x、y 各 U(−θ, θ) 的傾斜（rad），推力方向跟著傾斜
    thrust_eps: float = 0.0            # k_j ~ U(1 − ε, 1 + ε)
    payload_event: dict = None         # 中途 payload 事件（None = 關閉）。見 ModularEnv 的「payload 事件」說明


# ============================================================================
# 3) 環境
# ============================================================================
class ModularEnv:
    def __init__(self, cells, n_envs, task="T2", device="cpu", seed=0, perturb=None, cfg=None):
        assert task in TASKS, f"task 必須是 {TASKS}"
        self.cfg = cfg or EnvConfig()
        if perturb is None: perturb = Perturb()
        elif isinstance(perturb, dict): perturb = Perturb(**perturb)
        self.perturb, self.task, self.B, self.device = perturb, task, n_envs, device
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.seed = seed
        self.ep = torch.zeros(n_envs, dtype=torch.long, device=device)      # 每個環境槽已經開始的 episode 編號（T2 目標序列用）
        dt_ = self.cfg.dtype
        self._tt = dict(dtype=dt_, device=device)

        # ---- 標稱組裝（policy 與古典法看到的形狀）----
        p, pitch, limits = load_module(self.cfg.props_path) if self.cfg.props_path else load_module()
        self.p, self.pitch, self.limits = p, pitch, limits
        self.cells = list(cells)
        self.nominal = assemble(self.cells, p, pitch)
        self.N = N = self.nominal["N"]
        self.nominal["c"] = np.c_[np.asarray(self.cells, float) * pitch, np.zeros(N)] - self.nominal["com"]  # 模組中心相對標稱 CoM
        self.n_rotors = 4 * N
        self.obs_dim = 3 + 3 + 6 + 3 + 3 + 2 + 4 + 4
        self.act_dim = 4
        self.adjacency = self._adjacency()                                 # [N, N, 4]
        self.neighbor_flags = self.adjacency.any(1).to(dt_)                # [N, 4]
        self.tmax, self.tau, self.km = p["Tmax"], p["tau"], p["km_ratio"]
        self.hover_goal = torch.tensor([0., 0., self.cfg.hover_height], **self._tt)

        # ---- 接頭：預先算好線性映射 L（外載 6N → 接頭 6E）----
        jm = build_joint_model(self.cells, p=p, pitch=pitch)
        self.n_joints = len(joint_edges(self.cells))
        self.joint_L = torch.as_tensor(joint_map(jm).reshape(self.n_joints * 6, 6 * N), **self._tt)  # [6E, 6N]
        lim = torch.tensor([limits["pull_N"], limits["shear_N"], limits["peel_moment_Nm"]], **self._tt)
        self.joint_limits = lim

        # ---- 狀態與幾何（先放標稱，reset 時依 perturb 重抽）----
        B, R4 = n_envs, 4 * N
        z = lambda *s: torch.zeros(*s, **self._tt)
        self.pos, self.vel, self.w = z(B, 3), z(B, 3), z(B, 3)
        self.q = z(B, 4); self.q[:, 0] = 1
        self.thrust = z(B, R4)                       # T_act
        self.u_prev = z(B, N, 4)
        self.goal_pos, self.goal_yaw = z(B, 3), z(B)
        self.t = torch.zeros(B, dtype=torch.long, device=device)
        self._F_b, self._alpha = z(B, 3), z(B, 3)    # 最近一個子步的機體系合力與角加速度（加速度計、接頭內力用）
        self.geo = self._sample_geometry(B)
        pt = self.perturb
        self._shared = not (pt.dock_delta > 0 or pt.dock_tilt > 0 or pt.thrust_eps > 0)   # payload 是固定的，不破壞「各環境相同」
        # ---- payload 事件（預設關閉；關閉時下面的東西都不會被碰到，env 輸出與沒有這個功能時逐位元相同）----
        self._events = pt.payload_event is not None
        self.ev_mode = torch.zeros(B, dtype=torch.long, device=device)      # 0 無事件、1 attach、2 drop
        self.t_event = torch.full((B,), 10 ** 9, dtype=torch.long, device=device)   # 事件發生的控制步（第 t_event 步的動作起用新幾何）
        self.ev_done = torch.zeros(B, dtype=torch.bool, device=device)
        self.ev_host = torch.zeros(B, dtype=torch.long, device=device)      # 事件的 payload 掛點模組（上限版控制器與診斷用，不是觀測）
        self.geo_alt = None
        if self._events:
            ev = pt.payload_event
            assert ev["type"] in ("attach", "drop") and ev.get("host", "random") in ("random", "argmax")
            assert pt.payload_frac == 0 and not (pt.dock_delta > 0 or pt.dock_tilt > 0 or pt.thrust_eps > 0), \
                "payload_event 不與 payload_frac、對接偏差、推力誤差並用"
            self._shared = False
            self.geo_alt = {k: v.clone() for k, v in self.geo.items()}
        self.reset()

    # ------------------------------------------------------------------ 幾何
    def _adjacency(self):
        """adjacency[i, j, k] = 模組 j 是模組 i 在方向 k 的鄰居；k = +x, −x, +y, −y。"""
        idx = {c: i for i, c in enumerate(self.cells)}
        adj = torch.zeros(self.N, self.N, 4, dtype=torch.bool)
        for c, i in idx.items():
            for k, (dx, dy) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
                j = idx.get((c[0] + dx, c[1] + dy))
                if j is not None: adj[i, j, k] = True
        return adj.to(self.device)

    def _sample_geometry(self, n):
        """n 個環境的真實幾何（payload、對接偏差、推力係數誤差）。回傳 dict，張量第 0 維 = n。
        M 質量；Jinv 轉動慣量逆；d 模組中心相對真實 CoM；J0 各模組（含傾斜）自身慣量；
        tdir 各轉子推力方向；tq 每單位推力產生的力矩（含反扭矩），即 r×dir + km·spin·dir；kmax 推力上限。"""
        tt, pt, N, m = self._tt, self.perturb, self.N, self.p["mass"]
        def U(*shape, a):                              # U(−a, a)
            return (torch.rand(*shape, generator=self.gen, device=self.device, dtype=tt["dtype"]) * 2 - 1) * a
        c_nom = torch.as_tensor(np.c_[np.asarray(self.cells, float) * self.pitch, np.zeros(N)], **tt)   # [N,3]
        c = c_nom + U(n, N, 3, a=pt.dock_delta)                                 # 真實模組中心 [n,N,3]
        Rt = rot_xy(U(n, N, a=pt.dock_tilt), U(n, N, a=pt.dock_tilt))            # 每個模組的傾斜 [n,N,3,3]
        masses = torch.full((n, N), m, **tt)
        com_num = (masses[..., None] * c).sum(1); M = masses.sum(1)
        mp, pos_p = 0.0, None
        if pt.payload_frac > 0:                                                 # payload：質點
            mp = pt.payload_frac / (1 - pt.payload_frac) * N * m
            host = pt.payload_host % N
            pos_p = c[:, host] + torch.tensor(pt.payload_offset, **tt)
            com_num = com_num + mp * pos_p; M = M + mp
        com = com_num / M[:, None]
        d = c - com[:, None]                                                    # [n,N,3]
        J0 = torch.diag(torch.tensor([self.p["Ixx"], self.p["Iyy"], self.p["Izz"]], **tt))
        J0r = Rt @ J0 @ Rt.transpose(-1, -2)                                    # [n,N,3,3]
        eye = torch.eye(3, **tt)
        par = lambda mass, dd: mass * ((dd * dd).sum(-1)[..., None, None] * eye - dd[..., :, None] * dd[..., None, :])
        J = (J0r + par(m, d)).sum(1)                                            # 平行軸定理
        if mp > 0: J = J + par(mp, pos_p - com)
        # 轉子：位置 = 模組中心 + 傾斜後的局部位置；推力方向 = 傾斜後的 ẑ
        loc = torch.as_tensor(np.c_[MOTOR_XY, np.zeros(4)], **tt)               # [4,3]
        rpos = d[:, :, None] + torch.einsum("nmij,rj->nmri", Rt, loc)           # [n,N,4,3]
        tdir = Rt[..., 2][:, :, None].expand(n, N, 4, 3)
        spin = torch.as_tensor(MOTOR_SPIN, **tt)[:, None]                        # [4,1]
        tq = torch.cross(rpos, tdir, dim=-1) + self.km * spin * tdir
        k = 1 + U(n, 4 * N, a=pt.thrust_eps)
        return dict(M=M, J=J, Jinv=torch.linalg.inv(J), d=d, J0=J0r,
                    tdir=tdir.reshape(n, 4 * N, 3), tq=tq.reshape(n, 4 * N, 3), kmax=self.tmax * k,
                    mp=torch.full((n,), mp, **tt), dp=(pos_p - com) if mp > 0 else torch.zeros(n, 3, **tt))

    # ------------------------------------------------------------------ 動力學
    def _terms(self, thrust, w):
        """給定推力與角速度，回傳機體系合力 F_b、合力矩 τ（對 CoM）與角加速度 α。"""
        g = self.geo
        if self._shared:        # 所有環境幾何相同（沒有隨機對接偏差／推力誤差）：一次矩陣乘法，比逐環境乘加快很多
            F_b, tau = thrust @ g["tdir"][0], thrust @ g["tq"][0]
            alpha = (tau - torch.cross(w, w @ g["J"][0].T, dim=-1)) @ g["Jinv"][0].T
            return F_b, tau, alpha
        F_b = (thrust[..., None] * g["tdir"]).sum(1)
        tau = (thrust[..., None] * g["tq"]).sum(1)
        alpha = matvec(g["Jinv"], tau - torch.cross(w, matvec(g["J"], w), dim=-1))
        return F_b, tau, alpha

    def _advance(self, thrust):
        """一個物理子步（不含馬達延遲）：半隱式 Euler；位置用更新後的速度，姿態乘 exp(ω_new dt)。"""
        dt = self.cfg.dt
        F_b, _, alpha = self._terms(thrust, self.w)
        R = quat_to_rot(self.q)
        acc = matvec(R, F_b) / self.geo["M"][:, None]
        acc = acc - torch.tensor([0., 0., G], **self._tt)                        # m dv = R F dt − m g ẑ dt
        self.vel = self.vel + acc * dt
        self.w = self.w + alpha * dt                                             # J dω = (τ − ω×Jω) dt
        self.pos = self.pos + self.vel * dt
        q = quat_mul(self.q, quat_exp(self.w, dt))
        self.q = q / q.norm(dim=-1, keepdim=True)
        self._F_b, self._alpha = F_b, alpha

    def _substep(self, T_cmd):
        """馬達一階延遲並 clip 到 [0, Tmax·k_j]，再積分一步。"""
        self.thrust = self.thrust + (T_cmd - self.thrust) * (self.cfg.dt / self.tau)
        self.thrust = torch.minimum(self.thrust.clamp_min(0), self.geo["kmax"])
        self._advance(self.thrust)

    # ------------------------------------------------------------------ 接頭內力
    def module_loads(self):
        """每個模組的 d'Alembert 外載 [B, N, 6]（機體系；參考點 = 該模組真實中心）。
        機體系下重力與慣性力中的 g 互相抵消，所以：F_i = 自己的推力 − m_i (F_b/M + α×d_i + ω×(ω×d_i))。
        M_i = 自己推力對模組中心的力矩（含反扭矩）− J0_i α − ω×J0_i ω。payload 的慣性力併入掛載模組。"""
        g, N, m = self.geo, self.N, self.p["mass"]
        w, al, B = self.w[:, None], self._alpha[:, None], self.B
        Tw = self.thrust[..., None]
        Tvec = (Tw * g["tdir"]).reshape(B, N, 4, 3).sum(2)                        # [B,N,3]
        Tmom = (Tw * g["tq"]).reshape(B, N, 4, 3).sum(2) - torch.cross(g["d"], Tvec, dim=-1)
        a_spec = self._F_b[:, None] / g["M"][:, None, None] + torch.cross(al, g["d"], dim=-1) \
            + torch.cross(w, torch.cross(w, g["d"], dim=-1), dim=-1)
        F = Tvec - m * a_spec
        mv = lambda A, x: (A * x[:, :, None]).sum(-1)                              # [B,N,3,3]·[B,N,3]
        wN = w.expand(B, N, 3)
        Mm = Tmom - mv(g["J0"], al.expand(B, N, 3)) - torch.cross(wN, mv(g["J0"], wN), dim=-1)
        if self.perturb.payload_frac > 0:
            host = self.perturb.payload_host % N
            dp = g["dp"]
            Fp = -g["mp"][:, None] * (self._F_b / g["M"][:, None] + torch.cross(self._alpha, dp, dim=-1)
                                      + torch.cross(self.w, torch.cross(self.w, dp, dim=-1), dim=-1))
            F = F.clone(); Mm = Mm.clone()
            F[:, host] += Fp
            Mm[:, host] += torch.cross(dp - g["d"][:, host], Fp, dim=-1)
        return torch.cat([F, Mm], -1)

    def joint_components(self):
        """每個接頭 6 個分量 [B, E, 6] = [f·n, f·t, f·ẑ, m·n, m·t, m·ẑ]（接頭座標，同 analysis/joint_loads.project）。"""
        ext = self.module_loads().reshape(self.B, 6 * self.N)
        return (ext @ self.joint_L.T).reshape(self.B, self.n_joints, 6)

    def joint_load_ratio(self):
        """[B]：max over 接頭 × {拉力, 剪力, 力矩} 的 負載/上限。
        拉力 = |f·n|；剪力 = hypot(f·t, f·ẑ)；力矩 = hypot(m·t, m·ẑ)（和 joint_loads 的 bend 同定義，扭矩 m·n 沒有上限不計）。"""
        if self.n_joints == 0: return torch.zeros(self.B, **self._tt)
        c = self.joint_components()
        r = torch.stack([c[..., 0].abs(), c[..., 1:3].norm(dim=-1), c[..., 4:6].norm(dim=-1)], -1) / self.joint_limits
        return r.amax((1, 2))

    # ------------------------------------------------------------------ 觀測
    def accelerometer(self):
        """各模組的比力 [B, N, 3]：R^T(a_c + g ẑ) + α×r_i + ω×(ω×r_i)；R^T(a_c+gẑ) = F_b/M（機體系）。
        r_i = 模組中心相對真實 CoM（policy 拿不到）。"""
        d, w, al = self.geo["d"], self.w[:, None], self._alpha[:, None]
        return self._F_b[:, None] / self.geo["M"][:, None, None] + torch.cross(al, d, dim=-1) \
            + torch.cross(w, torch.cross(w, d, dim=-1), dim=-1)

    def yaw(self):
        R = quat_to_rot(self.q)
        return torch.atan2(R[:, 1, 0], R[:, 0, 0])

    def observe(self, noise=True):
        c, B, N = self.cfg, self.B, self.N
        R = quat_to_rot(self.q); Rt = R.transpose(1, 2)
        e_p = matvec(Rt, self.goal_pos - self.pos)
        v_b = matvec(Rt, self.vel)
        att = torch.cat([R[:, :, 0], R[:, :, 1]], -1)                            # R 的前兩欄
        e_yaw = wrap_pi(self.goal_yaw - self.yaw())
        gyro, acc = self.w, self.accelerometer()
        if noise:
            gyro = gyro + c.gyro_noise * torch.randn(gyro.shape, generator=self.gen, **self._tt)
            acc = acc + c.accel_noise * torch.randn(acc.shape, generator=self.gen, **self._tt)
        nav = torch.cat([e_p, v_b, att, gyro], -1)[:, None].expand(B, N, 15)       # 整架共用的導航狀態
        yaw_err = torch.stack([e_yaw.sin(), e_yaw.cos()], -1)[:, None].expand(B, N, 2)
        flags = torch.zeros_like(self.neighbor_flags) if c.blind else self.neighbor_flags
        return torch.cat([nav, acc, yaw_err, flags.expand(B, N, 4), self.u_prev], -1)

    # ------------------------------------------------------------------ reset
    def reset(self):
        self.ep.zero_()
        self._reset_idx(torch.arange(self.B, device=self.device))
        return self.observe()

    def _reset_idx(self, idx):
        c, n, tt = self.cfg, len(idx), self._tt
        rnd = lambda *s, a=1.0: (torch.rand(*s, generator=self.gen, **tt) * 2 - 1) * a
        # 幾何（只有有誤差時才需要重抽）
        pt = self.perturb
        if pt.dock_delta > 0 or pt.dock_tilt > 0 or pt.thrust_eps > 0:
            new = self._sample_geometry(n)
            if idx.numel() == self.B: self.geo = new
            else:
                for k in new: self.geo[k][idx] = new[k]
        if self.perturb.payload_event is not None: self._reset_events(idx)          # 事件模式：幾何設成「事件前」那一份
        goal = self.hover_goal.expand(n, 3).clone(); gyaw = torch.zeros(n, **tt)
        pos = goal.clone(); vel = torch.zeros(n, 3, **tt); w = torch.zeros(n, 3, **tt)
        roll = torch.zeros(n, **tt); pitch = torch.zeros(n, **tt); yaw = torch.zeros(n, **tt)
        if self.task == "T1":
            pos = goal + rnd(n, 3, a=c.t1_pos_max); vel = rnd(n, 3, a=c.t1_vel_max); w = rnd(n, 3, a=c.t1_rate_max)
            roll, pitch, yaw = rnd(n, a=c.t1_tilt_max), rnd(n, a=c.t1_tilt_max), rnd(n, a=math.pi)
        elif self.task == "T2":
            goal, gyaw = self._t2_goal(idx, torch.zeros_like(idx))
        elif self.task == "ATT_STEP":
            mag = c.att_step[0] + torch.rand(n, generator=self.gen, **tt) * (c.att_step[1] - c.att_step[0])
            mag = mag * torch.where(torch.rand(n, generator=self.gen, **tt) < 0.5, -1.0, 1.0)
            on_roll = torch.rand(n, generator=self.gen, **tt) < 0.5
            roll, pitch = torch.where(on_roll, mag, 0 * mag), torch.where(on_roll, 0 * mag, mag)
        elif self.task == "POS_STEP":
            ang = torch.rand(n, generator=self.gen, **tt) * 2 * math.pi
            goal = goal + c.pos_step * torch.stack([ang.cos(), ang.sin(), 0 * ang], -1)
        self.pos[idx], self.vel[idx], self.w[idx] = pos, vel, w
        self.q[idx] = quat_from_euler(roll, pitch, yaw)
        self.goal_pos[idx], self.goal_yaw[idx], self.t[idx] = goal, gyaw, 0
        hover = self.geo["M"][idx] * G / self.n_rotors                           # 每顆約 Mg/(4N)
        self.thrust[idx] = hover[:, None].expand(n, self.n_rotors)
        self.u_prev[idx] = (2 * hover / self.tmax - 1)[:, None, None].expand(n, self.N, 4)
        F_b, _, alpha = self._terms(self.thrust, self.w)                         # 為了加速度計的初值
        self._F_b[idx], self._alpha[idx] = F_b[idx], alpha[idx]

    # ------------------------------------------------------------------ payload 事件
    # perturb.payload_event = dict(type="attach"|"drop", t_range=(1.0, 3.5), frac=0.15, host="random"|"argmax", offset=None|(x,y,z))
    #   attach：事件前沒有 payload，事件時刻起有；drop：相反（episode 一開始就帶著 payload，事件時刻掉掉）。
    #   事件時刻 = U(t_range) 秒，每個環境每個 episode 抽一次，換算成控制步 t_event（第 t_event 步的動作起用新幾何）。
    #   事件時刻與 random 掛點用無狀態雜湊亂數（只由 seed、環境槽、episode 編號決定）：同 seed 下所有方法成對，且不吃 self.gen（不影響其他亂數）。
    #   事件時質量、CoM、慣量、轉子相對 CoM 位置立即換成另一份幾何。連續性的處理（最簡單的做法）：
    #   把整架（含 payload）當同一個剛體，事件前後角速度 w 不變、剛體上每個質點的速度場不變——
    #   原點（真實 CoM）從 p 移到 p' = p + R s（s = 機體系的 CoM 位移 = d_舊 − d_新），新 CoM 的速度 = 舊速度場在該點的值 v + R (w × s)，
    #   所以 payload 以「當下掛點的機體速度」加入／離開，不做動量守恆的衝擊修正（沒有速度跳變，也不模擬掛上的衝量）。
    #   世界座標的目標不動，所以事件時刻位置誤差會有 |s| 大小的跳變（line 構型 15% 約 1–4 cm 量級）。
    #   限制：不與 payload_frac、對接偏差、推力誤差並用；事件期間 module_loads（接頭負載）不含 payload，只影響接頭量測。
    def _reset_events(self, idx):
        ev, n, N = self.perturb.payload_event, len(idx), self.N
        z = torch.zeros_like(idx)
        lo, hi = ev.get("t_range", (1.0, 3.5))
        dtc = self.cfg.dt * self.cfg.substeps
        t_ev = torch.floor((lo + (hi - lo) * self._hash_uniform(idx, z, 7).double()) / dtc).long()
        if ev.get("host", "random") == "argmax":
            hosts = torch.full((n,), int(np.argmax([c[0] for c in self.cells])), dtype=torch.long, device=self.device)
        else:
            hosts = (self._hash_uniform(idx, z, 8) * N).long().clamp_max(N - 1)
        pay = self._payload_geo(hosts, ev["frac"], ev.get("offset"))
        non = self._payload_geo(hosts, 0.0, None)
        attach = ev["type"] == "attach"
        pre, post = (non, pay) if attach else (pay, non)
        self._set_event_geo(idx, pre, post)
        self.ev_mode[idx] = 1 if attach else 2
        self.t_event[idx], self.ev_done[idx], self.ev_host[idx] = t_ev, False, hosts

    def _payload_geo(self, hosts, frac, offset):
        """逐環境 host 的 payload 幾何 [n,...]（無對接偏差／推力誤差，所以只依 host 不同；同 host 的環境一次算）。frac = 0 → 標稱。"""
        base, parts, order = self.perturb, [], []
        try:
            for h in hosts.unique().tolist():
                m = (hosts == h).nonzero()[:, 0]
                self.perturb = dataclasses.replace(base, payload_event=None, payload_frac=float(frac), payload_host=int(h),
                                                   payload_offset=tuple(offset) if offset is not None else (0., 0., 0.))
                parts.append(ModularEnv._sample_geometry(self, len(m))); order.append(m)
        finally:
            self.perturb = base
        order = torch.cat(order)
        inv = torch.empty_like(order); inv[order] = torch.arange(len(order), device=order.device)
        return {k: torch.cat([p[k] for p in parts], 0)[inv] for k in parts[0]}

    def _set_event_geo(self, idx, pre, post):
        """idx 的環境：目前幾何 = pre（事件前），另存 post（事件後）。"""
        if self.geo_alt is None: self.geo_alt = {k: v.clone() for k, v in self.geo.items()}
        for k in pre:
            self.geo[k][idx] = pre[k].to(self.geo[k].dtype)
            self.geo_alt[k][idx] = post[k].to(self.geo[k].dtype)

    def _apply_events(self):
        """step 開頭呼叫：t == t_event 的環境把幾何換成事件後那份，並平移原點（見上方說明）。"""
        fire = ((self.t == self.t_event) & ~self.ev_done & (self.ev_mode > 0)).nonzero()[:, 0]
        if not fire.numel(): return
        g, ga = self.geo, self.geo_alt
        s = g["d"][fire, 0] - ga["d"][fire, 0]                                   # 機體系 CoM 位移（d = c − CoM，所以 s = CoM_新 − CoM_舊）
        R = quat_to_rot(self.q[fire])
        Rs = matvec(R, s)
        self.pos[fire] = self.pos[fire] + Rs
        self.vel[fire] = self.vel[fire] + torch.cross(matvec(R, self.w[fire]), Rs, dim=-1)
        for k in g:
            tmp = g[k][fire].clone(); g[k][fire] = ga[k][fire]; ga[k][fire] = tmp
        F_b, _, alpha = self._terms(self.thrust, self.w)
        self._F_b[fire], self._alpha[fire] = F_b[fire], alpha[fire]
        self.ev_done[fire] = True

    def _hash_uniform(self, env_idx, seg, stream):
        """無狀態的雜湊亂數 U[0,1)：只由 (seed, 環境槽, episode 編號, 段, stream) 決定（splitmix64 風格，int64 溢位回繞）。
        T2 的目標序列用它，所以不管別的 episode 何時終止，同一個 (槽, episode 編號) 的目標序列都一樣。"""
        mask = lambda k: (1 << (64 - k)) - 1                                     # 模擬無號右移
        def s64(c): return c - (1 << 64) if c >= (1 << 63) else c
        x = (s64((self.seed * 0x9E3779B97F4A7C15 + 0x1234567) & ((1 << 64) - 1)) + (env_idx + 1) * s64(0xBF58476D1CE4E5B9)
             + (self.ep[env_idx] + 1) * s64(0x94D049BB133111EB) + (seg + 1) * s64(0xD6E8FEB86659FD93) + (stream + 1) * s64(0xFF51AFD7ED558CCD))
        for _ in range(2):
            x = x ^ ((x >> 30) & mask(30)); x = x * s64(0xBF58476D1CE4E5B9)
            x = x ^ ((x >> 27) & mask(27)); x = x * s64(0x94D049BB133111EB)
            x = x ^ ((x >> 31) & mask(31))
        return ((x >> 40) & ((1 << 24) - 1)).to(self.cfg.dtype) / float(1 << 24)

    def _t2_goal(self, idx, seg):
        """環境槽 idx 在「目前 episode、第 seg 段」的 T2 目標（位置、yaw）。逐 episode 成對，與其他環境無關。"""
        c = self.cfg
        u = lambda k: self._hash_uniform(idx, seg, k) * 2 - 1
        g = self.hover_goal + torch.stack([u(0), u(1), u(2)], -1) * c.t2_pos_max
        return g, u(3) * c.t2_yaw_max

    def set_state(self, idx=None, **kw):
        """直接設定狀態（測試與古典控制器評估用）。可設 pos、vel、q、w、thrust、goal_pos、goal_yaw。"""
        idx = slice(None) if idx is None else idx
        for k, v in kw.items():
            getattr(self, k)[idx] = torch.as_tensor(v, **self._tt)
        F_b, _, alpha = self._terms(self.thrust, self.w)
        self._F_b[idx], self._alpha[idx] = F_b[idx], alpha[idx]

    # ------------------------------------------------------------------ step
    def state(self):
        return dict(pos=self.pos, vel=self.vel, q=self.q, w=self.w, thrust=self.thrust,
                    goal_pos=self.goal_pos, goal_yaw=self.goal_yaw, t=self.t, u_prev=self.u_prev)

    def step(self, action):
        c, N = self.cfg, self.N
        if self._events: self._apply_events()
        u = action.to(**self._tt).clamp(-1, 1)
        T_cmd = (self.tmax * (u + 1) / 2).reshape(self.B, 4 * N)
        for _ in range(c.substeps): self._substep(T_cmd)
        self.t += 1
        if self.task == "T2":                                                    # 每 1.5 s 換目標
            sw = (self.t % c.t2_period == 0).nonzero()[:, 0]
            if sw.numel():
                self.goal_pos[sw], self.goal_yaw[sw] = self._t2_goal(sw, self.t[sw] // c.t2_period)

        ratio = self.joint_load_ratio()
        R = quat_to_rot(self.q)
        e_p = self.goal_pos - self.pos
        e_yaw = wrap_pi(self.goal_yaw - self.yaw())
        tilt = torch.acos(R[:, 2, 2].clamp(-1, 1))
        overload = (ratio > 1) & c.detach_on_overload
        terminated = (self.pos[:, 2] < 0) | (e_p.norm(dim=-1) > c.max_pos_err) | (tilt > c.max_tilt) | overload
        truncated = (self.t >= c.episode_steps) & ~terminated

        du2 = ((u - self.u_prev) ** 2).mean((1, 2))
        reward = -(c.w_pos * e_p.norm(dim=-1) + c.w_vel * self.vel.norm(dim=-1) + c.w_rate * self.w.norm(dim=-1)
                   + c.w_yaw * e_yaw.abs()) - c.w_action * du2 + c.alive \
            - c.terminal * terminated.to(self.cfg.dtype) - c.w_joint * ratio
        self.u_prev = u

        info = dict(joint_load_ratio=ratio, overload=overload)
        obs = self.observe()
        info["final_obs"] = obs                                                  # SAME_STEP：reset 之前的 obs
        info["final_state"] = {k: v.clone() for k, v in self.state().items()}
        done = (terminated | truncated).nonzero()[:, 0]
        if done.numel():
            self.ep[done] += 1
            self._reset_idx(done)
            obs = obs.clone()
            obs[done] = self.observe()[done]
        info["state"] = self.state()                                             # reset 之後的狀態，和回傳的 obs 對應
        return obs, reward, terminated, truncated, info
