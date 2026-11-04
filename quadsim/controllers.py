"""古典控制 baseline（P2 / G1）：同一個外迴路，差別只在「分配層」。

外迴路（所有方法共用，Lee et al. 2010 的 SE(3) 幾何控制，增益用「正規化」形式）：
  位置 PD → 期望加速度 a_des → 期望推力方向 b3d 與期望姿態 R_d（含 yaw）
  姿態誤差 e_R、角速度誤差 e_ω → 期望角加速度 α_cmd = −kR·e_R − kω·e_ω
  位置用加速度、姿態用角加速度，所以增益與質量、慣量無關，同一組可以用在所有構型。
外迴路輸出兩個量，交給分配層：
  f_acc = |a_des + g ẑ|（每單位質量要的推力，m/s²）與 α_cmd（rad/s²）。

分配層（差別只在這裡）：
  B1      oracle：知道完整標稱形狀，τ = Jα + ω×Jω、F = M f_acc，用整架 A 的 pinv 分配，再 clip。
  B1-QP   同樣的 wrench，改用有界最小平方分配；飽和時依優先序犧牲（roll/pitch > Fz > yaw），
          可選接頭負載懲罰。
  B2(k)   古典 k 跳規則：每個模組只用 k 跳球內的形狀（joint_loads.alloc_khop 的定義）；
          每個模組出自己那份 Fz = m f_acc。
  B2+adapt  B2 加上逐軸增益適應（帶遺忘因子的 RLS）。
  B2-QP(+adapt)  B2 的 k 跳 wrench 估計 + 球內有界最小平方分配（處理推力飽和）。
  B5      天真：每個模組都以為自己是單機。

統一介面：ctrl.act(state, goal, env) -> action [B, N, 4]，範圍 [−1, 1]，和 env 的動作定義相同。
  state = env.step 回傳的 info["state"]（或 env.state()）；goal = dict(pos=[B,3], yaw=[B])（用 goal_of(state) 取）。
  控制器只用「標稱」形狀（env.nominal、env.cells）；env 的真實幾何（payload、對接偏差）它看不到。

檔案由上到下：1) 增益與外迴路  2) 分配層基底  3) B1、B1-QP  4) B2、B2+adapt  5) B5  6) 工廠函式
"""
from dataclasses import dataclass, asdict
import numpy as np
import torch
from scipy.optimize import lsq_linear

from analysis.joint_loads import build as build_joint_model, alloc_khop, alpha_max, module_loads, hop_ball
from .assembly import assemble, MOTOR_XY, MOTOR_SPIN
from .env import quat_to_rot
from .params import G, CF2X


# ============================================================================
# 1) 增益與外迴路
# ============================================================================
@dataclass
class Gains:
    """正規化增益。kp、kd 作用在位置（輸出 m/s²）；kR、kw 作用在姿態（輸出 rad/s²）。"""
    kp: float = 8.0          # 1/s²   位置誤差 → 加速度
    kd: float = 5.0          # 1/s    速度誤差 → 加速度
    kR: float = 100.0        # 1/s²   姿態誤差 e_R → 角加速度
    kw: float = 15.0         # 1/s    角速度誤差 e_ω → 角加速度
    yaw_ratio: float = 0.3   # yaw 軸的 kR、kw 是 roll/pitch 的幾倍（yaw 的力矩權限小很多，同一個增益會一直飽和）
    a_xy_max: float = 0.5 * G    # m/s²  水平加速度上限（固定值，不調），避免位置誤差大時要求過度傾斜
    alpha_frac: float = 0.8      # α_cmd 每軸上限 = alpha_frac × 該構型 oracle 的無飽和最大 α（固定值，不調；見 outer_loop）
    acc_z_min: float = 0.5 * G   # m/s²  推力加速度的垂直分量下限（固定值，不調）：降得太低，馬達貼近 0，就沒有差動力矩可用

    def to_dict(self): return asdict(self)


def goal_of(state):
    """從 env 的 state 取出 goal dict（act 的第二個參數）。"""
    return dict(pos=state["goal_pos"], yaw=state["goal_yaw"])


def vee(S):
    """反對稱矩陣 [B,3,3] → 向量 [B,3]。"""
    return torch.stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]], -1)


def outer_loop(g, state, goal, alpha_lim=None):
    """共用外迴路。alpha_lim [3]：期望角加速度每軸的上限（rad/s²）；None = 不限幅。
    為什麼要限幅：正規化增益讓同一組增益能用在所有構型，但各構型的角加速度權限差 25 倍（P0）；
      大構型的權限小，姿態誤差一大，指令遠超過馬達做得到的，推力一飽和 pinv 解就一起被 clip，整架失去控制。
      這是外迴路的保護（所有方法同一個限幅），不是分配層的一部分。限幅值取自標稱形狀的 oracle 權限，
      所以 leader 多知道三個數字（總體權限），這比 B2 的「只看 k 跳」多，對古典法有利。
    回傳 dict：f_acc [B]、alpha_cmd [B,3]、R_d [B,3,3]、att_err [B]（R 與 R_d 的夾角，rad）。"""
    pos, vel, w = state["pos"], state["vel"], state["w"]
    R = quat_to_rot(state["q"])
    gz = torch.tensor([0., 0., G], dtype=pos.dtype, device=pos.device)

    # --- 位置 PD → 期望加速度（水平限幅）→ 期望推力方向 ---
    a_des = g.kp * (goal["pos"] - pos) - g.kd * vel
    a_xy = a_des[:, :2]
    scale = (g.a_xy_max / a_xy.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
    a_des = torch.cat([a_xy * scale, a_des[:, 2:]], -1)
    acc = a_des + gz                                            # 要用推力抵消的加速度（含重力）
    acc = torch.cat([acc[:, :2], acc[:, 2:].clamp_min(g.acc_z_min)], -1)   # 垂直分量留下限（也避免 b3d 退化）
    f_acc = acc.norm(dim=-1)
    b3d = acc / f_acc[:, None]

    # --- 期望姿態：b3d 給定，再用期望 yaw 決定另外兩軸 ---
    yaw_d = goal["yaw"]
    xc = torch.stack([yaw_d.cos(), yaw_d.sin(), torch.zeros_like(yaw_d)], -1)
    b2d = torch.cross(b3d, xc, dim=-1)
    b2d = b2d / b2d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    b1d = torch.cross(b2d, b3d, dim=-1)
    R_d = torch.stack([b1d, b2d, b3d], -1)

    # --- 姿態誤差 → 期望角加速度（ω_d = 0，所以 e_ω = ω）---
    Rt = R.transpose(1, 2)
    e_R = 0.5 * vee(R_d.transpose(1, 2) @ R - Rt @ R_d)
    axis = torch.tensor([1., 1., g.yaw_ratio], dtype=pos.dtype, device=pos.device)      # roll、pitch、yaw 的增益倍率
    alpha_cmd = axis * (-g.kR * e_R - g.kw * w)
    if alpha_lim is not None: alpha_cmd = torch.maximum(torch.minimum(alpha_cmd, alpha_lim), -alpha_lim)
    cos = ((R_d * R).sum((1, 2)) - 1) / 2                       # tr(R_dᵀ R) = Σ R_d ⊙ R
    return dict(f_acc=f_acc, alpha_cmd=alpha_cmd, R_d=R_d, att_err=torch.acos(cos.clamp(-1, 1)))


# ============================================================================
# 2) 分配層基底
# ============================================================================
class Classical:
    """子類只需實作 _setup（用標稱形狀預先算矩陣）與 _thrusts（f_acc、α_cmd → 每顆轉子推力 [B,4N]，未 clip）。"""
    name = "base"

    def __init__(self, gains=None):
        self.gains = gains or Gains()
        self._env = None

    def attitude_error(self, state, goal):
        """外迴路的姿態誤差（R 與 R_d 的夾角），G1 用它量「姿態追蹤」。"""
        return outer_loop(self.gains, state, goal)["att_err"]

    def act(self, state, goal, env):
        if self._env is not env:                                # 第一次遇到這個 env：用標稱形狀預先算矩陣
            self._env = env
            nom = env.nominal
            self.N, self.M, self.J, self.m = env.N, nom["M"], nom["J"], env.p["mass"]
            self.J0 = np.diag([env.p["Ixx"], env.p["Iyy"], env.p["Izz"]])
            self.tt = dict(dtype=env.cfg.dtype, device=env.device)
            self.tmax = env.tmax
            self.mod = build_joint_model(env.cells, p=env.p, pitch=env.pitch)       # joint_loads 的整架模型（標稱）
            self.alpha_lim = self._tensor([self.gains.alpha_frac * self._amax(a, env) for a in range(3)])
            self._setup(env)
        o = outer_loop(self.gains, state, goal, self.alpha_lim)
        T_raw = self._thrusts(o, state, env)
        T = T_raw.clamp(0, self.tmax)                           # 馬達物理上限；env 端也會 clip，這裡先轉成合法動作
        self._T_last = T                                         # 實際送出的推力（INDI、Fz 適應估計器用）
        return (2 * T / self.tmax - 1).reshape(-1, self.N, 4)

    def _setup(self, env): pass

    def _amax(self, axis, env):
        """該構型 oracle 的無飽和最大 α；推力上限取 env.tmax（預設等於 CF2X 的值，結果不變；診斷放大推力上限時才不同）。"""
        return alpha_max(self.mod, axis, None if env.tmax == CF2X["Tmax"] else env.tmax)

    def _tensor(self, x): return torch.as_tensor(np.asarray(x), **self.tt)

    def _wrench_torque(self, alpha_cmd, w, J):
        """τ = Jα + ω×Jω（J 是 [3,3] 張量）。"""
        return alpha_cmd @ J.T + torch.cross(w, w @ J.T, dim=-1)


# ============================================================================
# 3) B1 oracle、B1-QP
# ============================================================================
class B1(Classical):
    """oracle：標稱的整架 J、A（完整形狀）。T = pinv(A)·[M f_acc, Jα + ω×Jω]。"""
    name = "B1"

    def _setup(self, env):
        self.pinvA = self._tensor(np.linalg.pinv(self._A(env)))            # [4N, 4]
        self.Jt = self._tensor(self.J)

    def _A(self, env): return env.nominal["A"]        # 分配矩陣（子類可換成別的形狀；預設 = 標稱）

    def _wrench(self, o, w):
        return torch.cat([(self.M * o["f_acc"])[:, None], self._wrench_torque(o["alpha_cmd"], w, self.Jt)], -1)

    def _thrusts(self, o, state, env):
        return self._wrench(o, state["w"]) @ self.pinvA.T


class B1QP(B1):
    """B1 的 wrench，改用有界最小平方分配（x = T/Tmax ∈ [0, 1]）：
        min  ‖W(A T − w)‖² + w_joint‖Jn T‖² + ridge‖x − x_ref‖²     s.t. 0 ≤ T ≤ Tmax
      W：wrench 四個分量先除以各自的最大可達值（讓權重可比），再乘優先序權重（roll/pitch > Fz > yaw）。
      Jn：接頭負載（除以上限）對 T 的線性映射；w_joint = 0 時關閉。
      ridge：很小的正則，讓解唯一；x_ref = 均分 Fz。沒有飽和且 w_joint = 0 時，解 = B1 的 pinv 解。
    不飽和的環境一次矩陣乘法求閉式解；解超出界限的環境才逐個呼叫 lsq_linear。"""
    name = "B1-QP"

    def __init__(self, gains=None, w_att=10.0, w_fz=3.0, w_yaw=1.0, w_joint=0.0, ridge=1e-8):
        super().__init__(gains)
        self.weights = np.array([w_fz, w_att, w_att, w_yaw])             # 對應 [Fz, τx, τy, τz]
        self.w_joint, self.ridge = w_joint, ridge
        self.n_qp = 0                                                     # 診斷：走過 lsq_linear 的次數

    def _setup(self, env):
        super()._setup(env)
        A, N, tmax = self._A(env), self.N, env.tmax
        cap = np.abs(A).sum(1) * tmax / np.array([1, 2, 2, 2])           # 各 wrench 分量的最大可達值（粗估）
        self.S = self.weights / cap                                       # [4] 列縮放
        rows = [self.S[:, None] * A * tmax]                               # 作用在 x = T/Tmax
        if self.w_joint > 0 and env.n_joints > 0:
            rows.append(np.sqrt(self.w_joint) * self._joint_matrix(env) * tmax)
        n_r = sum(r.shape[0] for r in rows)
        rows.append(np.sqrt(self.ridge) * np.eye(4 * N))
        self.Mat = np.vstack(rows)
        self.n_rows = n_r
        self.Minv = np.linalg.pinv(self.Mat)                              # [4N, rows]

    def _joint_matrix(self, env):
        """接頭負載比（5 個分量：拉力、剪力×2、彎矩×2，各除以上限）對 T 的線性映射 [5E, 4N]。
        外載對 T 是線性的（ω=0 時），所以對 4N 個單位推力各呼叫一次 module_loads 就得到矩陣。"""
        mod = self.mod
        ET = np.stack([module_loads(mod, np.eye(4 * self.N)[j]).reshape(-1) for j in range(4 * self.N)], 1)  # [6N,4N]
        L = env.joint_L.double().numpy() @ ET                             # [6E, 4N]，每接頭 [f·n, f·t, f·ẑ, m·n, m·t, m·ẑ]
        lim = env.joint_limits.double().numpy()                           # [拉, 剪, 彎矩]
        pick = [(0, lim[0]), (1, lim[1]), (2, lim[1]), (4, lim[2]), (5, lim[2])]
        return np.vstack([L[e * 6 + c] / l for e in range(env.n_joints) for c, l in pick])

    def _thrusts(self, o, state, env):
        wr = self._wrench(o, state["w"])
        return self._solve(wr, wr[:, 0:1] / (4 * self.N * self.tmax))      # x_ref = 均分 Fz

    def _solve(self, wr, x_ref):
        """有界最小平方：wr [B,4]（要達成的 wrench）、x_ref [B,1] 或 [B,4N]（ridge 拉向的 x = T/Tmax）。回傳 T [B,4N]。"""
        wr, x_ref = wr.double().numpy(), x_ref.double().numpy()
        B = wr.shape[0]
        rhs = np.zeros((B, self.Mat.shape[0]))
        rhs[:, :4] = wr * self.S
        rhs[:, self.n_rows:] = np.sqrt(self.ridge) * x_ref
        x = rhs @ self.Minv.T                                             # 無界閉式解
        bad = np.where(((x < -1e-9) | (x > 1 + 1e-9)).any(1))[0]
        for b in bad:                                                     # 飽和的環境才解有界問題
            x[b] = lsq_linear(self.Mat, rhs[b], bounds=(0.0, 1.0), method="bvls").x
        self.n_qp += len(bad)
        self._sat = np.zeros(B, bool); self._sat[bad] = True
        return self._tensor(x * self.tmax)


class B1QPInformed(B1QP):
    """B1-QP-informed（G2 特權 oracle）：和 B1-QP 一樣，但 M、J、A（轉子相對真實 CoM 的位置）
    改用「含 payload 的真實」模型（質點 payload 掛在模組 payload_host 的中心）。用來量「不知道 payload」的代價。
    外迴路 α 限幅仍用標稱形狀的值（和所有方法一樣；改用含 payload 的 alpha_max 會把限幅壓到 1/5～1/6，
    line3 payload15 的成功率從 100% 掉到 28%，那是限幅設計的副作用，不是資訊的價值）。
    只知道 payload；對接偏差與推力係數誤差仍是未知（和其他方法一樣用標稱）。payload_frac = 0 時與 B1-QP 完全相同。"""
    name = "B1-QP-informed"

    def __init__(self, gains=None, payload_frac=0.0, payload_host=-1, **kw):
        super().__init__(gains, **kw)
        self.payload_frac, self.payload_host = payload_frac, payload_host

    def _setup(self, env):
        N, m = env.N, env.p["mass"]
        mp = self.payload_frac / (1 - self.payload_frac) * N * m if self.payload_frac > 0 else 0.0
        self.true = build_joint_model(env.cells, payload=mp, host=self.payload_host % N, p=env.p, pitch=env.pitch)
        self.M, self.J = self.true["M"], self.true["J"]
        super()._setup(env)

    def _A(self, env): return self.true["A"]


def bounded_ls_batch(Mat, rhs, max_iter=25, tol=1e-9):
    """批次有界最小平方：對每一列 r 解 min ‖Mat x − r‖²  s.t. 0 ≤ x ≤ 1（x 的每個分量）。
    作用集法（固定超界的分量、對自由分量解正規方程、再依梯度符號釋放），所有列一起做批次線性代數，收斂的列即時移出；
    max_iter 內沒收斂的列（循環或病態）改用 lsq_linear(bvls) 兜底。回傳 [K, n]。"""
    K, n = rhs.shape[0], Mat.shape[1]
    H, Mtr_all = Mat.T @ Mat, rhs @ Mat                                 # [n,n]、[K,n]
    eye = np.eye(n)
    out = np.zeros((K, n))
    act = np.arange(K)
    fixed = np.zeros((K, n), bool); xfix = np.zeros((K, n)); Mtr = Mtr_all
    for it in range(max_iter):
        P = ~fixed
        Hm = H[None] * (P[:, :, None] & P[:, None, :]) + eye[None] * fixed[:, :, None]
        g = (Mtr - xfix @ H.T) * P
        x = np.where(P, np.linalg.solve(Hm, g[:, :, None])[:, :, 0], xfix)
        lo, hi = P & (x < 0), P & (x > 1)
        new = lo | hi
        grad = x @ H.T - Mtr
        wants = (fixed & (xfix == 0) & (grad < -tol)) | (fixed & (xfix == 1) & (grad > tol))   # 固定在界上但該往內走
        done = ~new.any(1) & ~wants.any(1)
        out[act[done]] = x[done]
        keep = ~done
        if not keep.any(): act = act[:0]; break
        act, x, new, lo, hi, wants, grad, fixed, xfix, Mtr = (v[keep] for v in (act, x, new, lo, hi, wants, grad, fixed, xfix, Mtr))
        rel = np.where(wants, np.abs(grad), 0.0).argmax(1)
        do_rel = ~new.any(1)
        xfix = np.where(lo, 0.0, np.where(hi, 1.0, xfix)); fixed = fixed | new
        r = np.where(do_rel)[0]
        fixed[r, rel[r]] = False; xfix[r, rel[r]] = 0.0
    for k in act:                                                         # 沒收斂：兜底
        out[k] = lsq_linear(Mat, rhs[k], bounds=(0.0, 1.0), method="bvls").x
    bounded_ls_batch.n_fallback = len(act)
    return np.clip(out, 0.0, 1.0)


# ============================================================================
# 4) B2 k 跳規則、B2+adapt
# ============================================================================
class B2(Classical):
    """每個模組只用 k 跳球內的形狀。分配定義直接用 analysis/joint_loads.alloc_khop：
      T_i = (hover 份額) + K_k·α_cmd；alloc_khop 對 α 是線性的，所以預先探出 K_k [4N, 3]。
    Fz：leader 廣播的是「每單位質量的推力 f_acc」，模組乘上自己的質量 m 即可，每顆轉子 m f_acc / 4，
      所以模組不需要知道 N（整架 Fz = N m f_acc = M f_acc，和 B1 相同）。
    不含陀螺項 ω×Jω（模組只知道球內形狀，這一項在本研究量級很小；若要補，是 B2 的改進項）。"""
    name = "B2"

    def __init__(self, k, gains=None):
        super().__init__(gains)
        self.k = k
        self.name = f"B2 k={k}"

    def _setup(self, env):
        base = alloc_khop(self.mod, np.zeros(3), self.k)
        K = np.stack([alloc_khop(self.mod, e, self.k) - base for e in np.eye(3)], 1)    # [4N, 3]
        self.K = self._tensor(K)

    def _alloc(self, f_acc, alpha):
        return (self.m * f_acc / 4)[:, None] + alpha @ self.K.T

    def _thrusts(self, o, state, env):
        return self._alloc(o["f_acc"], o["alpha_cmd"])


class B2Adapt(B2):
    """B2 + 逐軸增益適應。
    所有模組共用 leader 廣播的量測 ω，所以全架只有一組 ĝ_k（k = roll, pitch, yaw）。
    模型：實際 α_k = g_k · α_fed_k（α_fed = 實際餵給分配的指令 = α_cmd / ĝ，再過馬達一階延遲）。
    估計：帶遺忘因子的遞迴最小平方（RLS，純量形式）。
      Sxy ← λ·Sxy + x·y；Sxx ← λ·Sxx + x²；ĝ = (Sxy + ε)/(Sxx + ε)    （ε 把 ĝ 拉向 1，沒有激勵時不亂跑）
      y = 量測 α（ω 差分，並用標稱 J 扣掉陀螺項）；x = 經馬達延遲濾波的 α_fed。
    為什麼選 RLS 不選梯度法：梯度法的步長要配合 x² 的大小，而 x 在懸停與激烈機動之間差好幾個數量級、
      各構型的 α 量級也不同；RLS 的更新量自動依累積激勵縮放，不用為每個構型調步長。遺忘因子讓它能跟著變化的增益。
    保護：x、y 先除以該軸的 α_max 正規化（讓 ε 在各構型有相同意義）；推力飽和的那一步不更新
      （飽和時 α 不是 g·指令）；ĝ clip 在 [0.05, 20]。量測 ω 我們自己加陀螺儀雜訊（env 的 state 是無雜訊的）。"""

    def __init__(self, k, gains=None, lam=0.995, eps=0.01, g_range=(0.05, 20.0), seed=0, fz_adapt=False):
        super().__init__(k, gains)
        self.name = f"B2+adapt k={k}"
        self.lam, self.eps, self.g_range, self.seed = lam, eps, g_range, seed
        self.fz = FzAdapter() if fz_adapt else None                               # G2：選用的總推力增益估計（預設關，G1 行為不變）

    def _setup(self, env):
        super()._setup(env)
        B = env.B
        z = lambda: torch.zeros(B, 3, **self.tt)
        self.g_hat, self.Sxy, self.Sxx = torch.ones(B, 3, **self.tt), z(), z()
        self.fed_prev, self.x_f, self.w_prev = z(), z(), z()
        self.sat_prev = torch.zeros(B, dtype=torch.bool, device=env.device)
        self.scale = self._tensor([self._amax(a, env) for a in range(3)])      # 每軸正規化尺度
        self.Jt, self.Jinv = self._tensor(self.J), self._tensor(np.linalg.inv(self.J))
        self.dt = env.cfg.dt * env.cfg.substeps                                    # 控制週期
        self.beta = 1 - float(np.exp(-self.dt / env.tau))                          # 馬達延遲在一個控制週期內的追上比例
        self.noise = env.cfg.gyro_noise
        self.gen = torch.Generator(device=env.device).manual_seed(self.seed)
        if self.fz is not None: self.fz.setup(self, env)

    def _learn(self, state):
        """用「上一步餵進去的指令」與「這一步量到的 ω 變化」更新 ĝ。"""
        w_n = state["w"] + self.noise * torch.randn(state["w"].shape, generator=self.gen, **self.tt)
        fresh = (state["t"] == 0)[:, None]                                         # 新 episode：估計器歸零
        w_mid = 0.5 * (w_n + self.w_prev)
        y = (w_n - self.w_prev) / self.dt + torch.cross(w_mid, w_mid @ self.Jt.T, dim=-1) @ self.Jinv.T
        x_new = self.x_f + self.beta * (self.fed_prev - self.x_f)                  # 延遲濾波後的指令
        x = 0.5 * (self.x_f + x_new)
        use = (~self.sat_prev[:, None] & ~fresh).to(x.dtype)
        xn, yn = x / self.scale, y / self.scale
        self.Sxy = torch.where(fresh, 0.0, self.lam * self.Sxy + use * xn * yn)
        self.Sxx = torch.where(fresh, 0.0, self.lam * self.Sxx + use * xn * xn)
        self.g_hat = ((self.Sxy + self.eps) / (self.Sxx + self.eps)).clamp(*self.g_range)
        self.x_f = torch.where(fresh, 0.0, x_new)
        self.w_prev = w_n

    def _thrusts(self, o, state, env):
        self._learn(state)
        alpha_fed = o["alpha_cmd"] / self.g_hat                                    # 指令除以估計增益
        f_acc = o["f_acc"] if self.fz is None else o["f_acc"] / self.fz.update(state, env)
        T_raw = self._alloc(f_acc, alpha_fed)
        self.fed_prev = alpha_fed
        self.sat_prev = self._sat_flag(T_raw)
        return T_raw

    def _sat_flag(self, T_raw):
        return ((T_raw < 0) | (T_raw > self.tmax)).any(-1)


class B2QP(B2):
    """B2 的飽和版：每個模組照 B2 的方式，只用 k 跳球內（alloc_khop 的球，半徑 k+1）的形狀估計需要的 wrench
      （Fz = n·m·f_acc、τ = J_ball·α_cmd，J_ball 與轉子位置都相對球內質心），再用 B1-QP 的有界最小平方分配
      （同樣的優先序權重 roll/pitch > Fz > yaw、同樣的列縮放與 ridge）解出球內所有轉子，取自己的 4 顆。
      分配範圍選「球內轉子」而不是「只有自己的 4 顆」：
        1) 只有自己 4 顆時，要用 4 顆轉子達成整個球的力矩（槓桿臂在球的尺度），自己的 Fz 份額也不夠；
           只能把力矩按份額縮小，等於放棄 k 跳球帶來的形狀資訊，k 就沒有意義了。
        2) 沒有飽和時，球內解 = alloc_khop 的 pinv 解（Fz、τx、τy、τz 的列互相正交），所以 B2-QP 嚴格包含 B2，
           差別只在飽和時：球內轉子可以互相補償，並依優先序犧牲 yaw 與 Fz。
      每個模組各自解、只取自己的切片，互相不協調（去中心化的代價，和 B2 相同）。沒有陀螺項（同 B2）。
      沒有飽和時閉式解；球內解超出界限的 (環境, 模組) 才解有界問題（bounded_ls_batch，批次作用集法，不收斂的列兜底 bvls）。"""
    name = "B2-QP"

    def __init__(self, k, gains=None, w_att=10.0, w_fz=3.0, w_yaw=1.0, ridge=1e-8, **kw):
        super().__init__(k, gains, **kw)
        self.weights = np.array([w_fz, w_att, w_att, w_yaw])
        self.ridge = ridge
        self.n_qp = 0
        self.name = f"B2-QP{'+adapt' if isinstance(self, B2Adapt) else ''} k={k}"

    def _setup(self, env):
        super()._setup(env)
        km, m, tmax = env.p["km_ratio"], self.m, self.tmax
        ctr = np.c_[np.asarray(env.cells, float) * env.pitch, np.zeros(self.N)]
        self.ball = []                                                    # 每個模組：(球, 自己的切片起點, n, Jh, Mat, Minv, 列縮放 S)
        for i in range(self.N):
            ball = hop_ball(env.cells, i, self.k + 1); n = len(ball)
            rc = ctr[ball] - ctr[ball].mean(0)
            Jh = sum(self.J0 + m * (r @ r * np.eye(3) - np.outer(r, r)) for r in rc)
            rot = np.vstack([r + np.r_[xy, 0] for r in rc for xy in MOTOR_XY])
            A = np.vstack([np.ones(4 * n), rot[:, 1], -rot[:, 0], km * np.tile(MOTOR_SPIN, n)])
            cap = np.abs(A).sum(1) * tmax / np.array([1, 2, 2, 2])
            S = self.weights / cap
            Mat = np.vstack([S[:, None] * A * tmax, np.sqrt(self.ridge) * np.eye(4 * n)])
            self.ball.append((ball, ball.index(i) * 4, n, Jh, Mat, np.linalg.pinv(Mat), S))

    def _alloc(self, f_acc, alpha):
        B = f_acc.shape[0]
        fz, al = f_acc.double().numpy(), alpha.double().numpy()
        out = np.zeros((B, 4 * self.N))
        sat = np.zeros(B, bool)
        for i, (ball, s, n, Jh, Mat, Minv, S) in enumerate(self.ball):
            wr = np.concatenate([(n * self.m * fz)[:, None], al @ Jh.T], 1)             # [B, 4]
            rhs = np.zeros((B, Mat.shape[0]))
            rhs[:, :4] = wr * S
            rhs[:, 4:] = np.sqrt(self.ridge) * (wr[:, 0] / (4 * n * self.tmax))[:, None]
            x = rhs @ Minv.T
            bad = np.where(((x < -1e-9) | (x > 1 + 1e-9)).any(1))[0]
            if len(bad): x[bad] = bounded_ls_batch(Mat, rhs[bad])
            self.n_qp += len(bad); sat[bad] = True
            out[:, 4 * i:4 * i + 4] = x[:, s:s + 4] * self.tmax
        self._sat = torch.as_tensor(sat, device=f_acc.device)
        return self._tensor(out)

    def _sat_flag(self, T_raw): return self._sat


class B2QPAdapt(B2QP, B2Adapt):
    """B2-QP + 逐軸增益適應（同 B2+adapt 的 RLS；飽和的那一步不更新，飽和 = 該環境任一模組的有界問題有碰到界限）。"""


class B2QPCoord(B2QP):
    """B2-QP-coord：B2-QP 加上 leader 廣播的全域純量，協調飽和（RQ-A 的公平性關卡）。
    每回合（rounds 次，預設 1 次協商 + 1 次最終解）：
      1) 各模組照 B2-QP 解出自己球內的分配，回報 4 個純量 r_j = 達成的 wrench 分量 / 要求的分量（j = Fz、roll、pitch、yaw；
         夾在 [0,1]；要求太小時回報 1）。模組不回報形狀、不回報別人的資訊。
      2) leader 對所有模組取 min（reduce="min"）或平均（"mean"），得到 4 個純量 s = (s_Fz, s_roll, s_pitch, s_yaw)，廣播給所有模組。
      3) 各模組把自己的要求乘上 s（只乘 scale_axes 指定的軸）後重解。所有模組用同一組 s，所以不會各自在 yaw 上出力不一。
    模組收到的資訊 = 原本 B2 的（f_acc、α_cmd、自己 k 跳球的形狀）+ 每回合 4 個純量。leader 收到每個模組 4 個純量。
    沒有任何模組飽和時 r = 1、s = 1，輸出 = B2-QP（位元級相同）。self.last_broadcast：[B, rounds, 4]，供測試檢查廣播維度。"""
    name = "B2-QP-coord"

    VARIANTS = {"v1": dict(mode="axes", rounds=1),        # 每軸各自的全域縮放比（4 個純量）
                "v2": dict(mode="priority", rounds=2),    # 優先序：只縮 yaw，且 roll/pitch/Fz 任何一軸達不到就繼續縮 yaw
                "v3": dict(mode="common", rounds=2)}      # 保方向：roll/pitch/yaw 力矩共用同一個縮放比（Fz 不縮），2 回合

    def __init__(self, k, gains=None, rounds=1, reduce="min", scale_axes=(0, 1, 2, 3), s_floor=0.0, mode="axes", **kw):
        super().__init__(k, gains, **kw)
        self.rounds, self.reduce, self.scale_axes, self.s_floor, self.mode = rounds, reduce, tuple(scale_axes), s_floor, mode
        self.name = f"B2-QP-coord{'+adapt' if isinstance(self, B2Adapt) else ''} k={k}"
        self.last_broadcast = None

    def _solve(self, fz, al, s):
        """所有模組各自解一次。s [B,4]：leader 廣播的縮放。回傳 (out [B,4N], 每模組的達成比 [B,N,4], 飽和旗標 [B])。"""
        B = fz.shape[0]
        out = np.zeros((B, 4 * self.N))
        ratio = np.ones((B, self.N, 4))
        sat = np.zeros(B, bool)
        for i, (ball, st, n, Jh, Mat, Minv, S) in enumerate(self.ball):
            wr = np.concatenate([(n * self.m * fz)[:, None], al @ Jh.T], 1) * s                 # [B,4]
            rhs = np.zeros((B, Mat.shape[0]))
            rhs[:, :4] = wr * S
            rhs[:, 4:] = np.sqrt(self.ridge) * (wr[:, 0] / (4 * n * self.tmax))[:, None]
            x = rhs @ Minv.T
            bad = np.where(((x < -1e-9) | (x > 1 + 1e-9)).any(1))[0]
            if len(bad): x[bad] = bounded_ls_batch(Mat, rhs[bad])
            self.n_qp += len(bad); sat[bad] = True
            out[:, 4 * i:4 * i + 4] = x[:, st:st + 4] * self.tmax
            ach = (x @ Mat[:4].T) / S                                                           # 達成的 wrench（與 wr 同單位）
            cap = self.weights / S
            ok = np.abs(wr) > 0.02 * cap
            ratio[:, i] = np.where(ok, np.clip(ach / np.where(ok, wr, 1.0), 0.0, 1.0), 1.0)
        return out, ratio, sat

    def _alloc(self, f_acc, alpha):
        B = f_acc.shape[0]
        fz, al = f_acc.double().numpy(), alpha.double().numpy()
        s = np.ones((B, 4))
        out, ratio, sat = self._solve(fz, al, s)
        bc = []
        for _ in range(self.rounds):
            r = ratio.min(1) if self.reduce == "min" else ratio.mean(1)                          # leader 彙整：[B,4]
            s_new = np.ones((B, 4))
            if self.mode == "axes":
                for a in self.scale_axes: s_new[:, a] = np.clip(r[:, a] * s[:, a], self.s_floor, 1.0)
            elif self.mode == "priority":
                s_new[:, 3] = np.clip(r.min(1) * s[:, 3], self.s_floor, 1.0)
            else:
                s_new[:, 1:] = np.clip(r[:, 1:].min(1, keepdims=True) * s[:, 1:], self.s_floor, 1.0)
            s = s_new
            bc.append(s.copy())
            out, ratio, sat = self._solve(fz, al, s)
        self.last_broadcast = torch.as_tensor(np.stack(bc, 1))                                  # [B, rounds, 4]
        self._sat = torch.as_tensor(sat, device=f_acc.device)
        return self._tensor(out)


class B2QPCoordAdapt(B2QPCoord, B2Adapt):
    """B2-QP-coord + 逐軸增益適應（同 B2QPAdapt）。"""


class YawLimMixin:
    """yaw 參考速率限制（古典法的公平性補強）：目標 yaw 是 ±90° 的階躍時，yaw 力矩權限小的構型會一直貼界。
    把 yaw 參考改成以固定速率追向目標：速率上限 = rate_c · sqrt(α_yaw_lim)（α_yaw_lim = 該構型標稱 oracle 的 yaw 角加速度權限
    × alpha_frac；用標稱形狀算，是依構型 yaw 權限做的增益排程）。每個環境的 yaw_ref 在 t==0 設為目前 yaw。"""
    def __init__(self, *a, rate_c=1.0, **kw):
        super().__init__(*a, **kw)
        self.rate_c, self.yaw_ref = rate_c, None
        self.name = self.name + "-yawlim"

    def act(self, state, goal, env):
        q = state["q"]
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        first = self._env is not env
        out_alpha = None
        if first:                                                  # 先讓基底把 alpha_lim 算出來再用；這一步用目前 yaw 當參考
            self.yaw_ref = yaw.clone()
            g2 = dict(pos=goal["pos"], yaw=self.yaw_ref)
            a = super().act(state, g2, env)
            self.dpsi = self.rate_c * float(self.alpha_lim[2].sqrt()) * env.cfg.dt * env.cfg.substeps
            return a
        fresh = state["t"] == 0
        self.yaw_ref = torch.where(fresh, yaw, self.yaw_ref)
        err = (goal["yaw"] - self.yaw_ref + np.pi) % (2 * np.pi) - np.pi
        self.yaw_ref = self.yaw_ref + err.clamp(-self.dpsi, self.dpsi)
        return super().act(state, dict(pos=goal["pos"], yaw=self.yaw_ref), env)


class B1QPYawLim(YawLimMixin, B1QP): pass
class B2QPCoordYawLim(YawLimMixin, B2QPCoordAdapt): pass


# ============================================================================
# 4b) G2：物理誤差下的古典法（總推力增益估計、B1-QP+adapt、B3 INDI、B4 簡化版）
# ============================================================================
class FzAdapter:
    """總推力（collective）增益估計：質量誤差、推力係數誤差都表現成「同樣的推力指令，實際比力 = κ × 標稱預期」。
    模型：f_meas = κ · ΣT_motor / M_nom（f_meas = 加速度計的機體 z 比力，取所有模組的平均；ΣT_motor 用標稱馬達一階模型從送出的指令推得）。
    估計：帶遺忘因子的 RLS（過原點的純量形式）；用 κ̂ 把 f_acc 除掉。這是 B4（UFO）估計質量的最簡化對應。
    控制器只用加速度計（雜訊同 env 的 accel_noise）與自己送出的指令，不看 env 的真實質量。"""

    def __init__(self, lam=0.99, eps=0.5, k_range=(0.3, 3.0), seed=1):
        self.lam, self.eps, self.k_range, self.seed = lam, eps, k_range, seed

    def setup(self, ctrl, env):
        tt, B = ctrl.tt, env.B
        self.ctrl = ctrl
        self.beta = 1 - (1 - env.cfg.dt / env.tau) ** env.cfg.substeps               # 一個控制週期內馬達的追上比例
        self.hover = ctrl.M * G / (4 * ctrl.N)
        self.T_m = torch.full((B, 4 * ctrl.N), self.hover, **tt)
        self.Sxy, self.Sxx = torch.zeros(B, **tt), torch.zeros(B, **tt)
        self.kappa = torch.ones(B, **tt)
        self.gen = torch.Generator(device=env.device).manual_seed(self.seed)

    def update(self, state, env):
        c = self.ctrl
        fresh = state["t"] == 0
        T_prev = getattr(c, "_T_last", None)
        if T_prev is None: T_prev = torch.full_like(self.T_m, self.hover)
        T_m = torch.where(fresh[:, None], torch.full_like(self.T_m, self.hover), self.T_m + self.beta * (T_prev - self.T_m))
        self.T_m = T_m
        f_meas = env.accelerometer()[..., 2].mean(1) + env.cfg.accel_noise * torch.randn(fresh.shape, generator=self.gen, **c.tt)
        x, y = T_m.sum(1) / c.M / G, f_meas / G                                      # 除以 g：讓 eps 有固定意義
        self.Sxy = torch.where(fresh, 0.0, self.lam * self.Sxy + x * y)
        self.Sxx = torch.where(fresh, 0.0, self.lam * self.Sxx + x * x)
        self.kappa = ((self.Sxy + self.eps) / (self.Sxx + self.eps)).clamp(*self.k_range)
        return self.kappa


class B1QPAdapt(B1QP):
    """B1-QP + 線上自適應（G2）：
      (a) 逐軸（roll、pitch、yaw）角加速度增益 ĝ：實際 α = ĝ · α_fed（RLS，同 B2Adapt；吸收慣量、推力係數的整體誤差）
      (b) 總推力增益 κ̂（FzAdapter；吸收質量、推力係數的整體誤差）
      (c) bias=True 時（= B4）再加逐軸力矩偏置 b̂：實際 α = ĝ·α_fed + b̂·s（s = f_acc/g，偏置正比於推力）。
          CoM 偏移 r_c 對 roll/pitch 的效果是 τ = −r_c × F，正比總推力，所以 b̂ 就是 CoM 偏移（在 α 空間）的估計；
          yaw 軸的 b̂ 吸收左右推力係數不對稱造成的反扭矩偏置。
    2 參數 RLS（遺忘因子、拉向 (ĝ, b̂) = (1, 0) 的小正則）每軸獨立；x、y 先除以該軸 α_max。飽和的那一步不更新。"""
    name = "B1-QP+adapt"

    def __init__(self, gains=None, bias=False, lam=0.995, eps=0.01, g_range=(0.5, 2.0), b_range=1.0, seed=0, fz_adapt=True, **kw):
        super().__init__(gains, **kw)
        self.bias, self.lam, self.eps, self.g_range, self.b_range, self.seed = bias, lam, eps, g_range, b_range, seed
        self.fz = FzAdapter() if fz_adapt else None
        self.name = "B4" if bias else "B1-QP+adapt"

    def _setup(self, env):
        super()._setup(env)
        B = env.B
        z = lambda *s: torch.zeros(B, *s, **self.tt)
        self.g_hat, self.b_hat = torch.ones(B, 3, **self.tt), z(3)
        self.Sxx, self.Sxy, self.Sxs, self.Sss, self.Ssy = (z(3) for _ in range(5))
        self.fed_prev, self.x_f, self.w_prev = z(3), z(3), z(3)
        self.sat_prev = torch.zeros(B, dtype=torch.bool, device=env.device)
        self.scale = self._tensor([self._amax(a, env) for a in range(3)])
        self.Jinv = self._tensor(np.linalg.inv(self.J))
        self.dt = env.cfg.dt * env.cfg.substeps
        self.beta = 1 - float(np.exp(-self.dt / env.tau))
        self.noise = env.cfg.gyro_noise
        self.s_prev = torch.ones(B, **self.tt)
        self.gen = torch.Generator(device=env.device).manual_seed(self.seed)
        if self.fz is not None: self.fz.setup(self, env)

    def _learn(self, state):
        w_n = state["w"] + self.noise * torch.randn(state["w"].shape, generator=self.gen, **self.tt)
        fresh = (state["t"] == 0)[:, None]
        w_mid = 0.5 * (w_n + self.w_prev)
        y = (w_n - self.w_prev) / self.dt + torch.cross(w_mid, w_mid @ self.Jt.T, dim=-1) @ self.Jinv.T
        x_new = self.x_f + self.beta * (self.fed_prev - self.x_f)
        x = 0.5 * (self.x_f + x_new)
        use = (~self.sat_prev[:, None] & ~fresh).to(x.dtype)
        xn, yn, s = x / self.scale, y / self.scale, self.s_prev[:, None].expand(-1, 3)
        up = lambda S, v: torch.where(fresh, 0.0, self.lam * S + use * v)
        self.Sxx, self.Sxy, self.Sxs, self.Sss, self.Ssy = (up(self.Sxx, xn * xn), up(self.Sxy, xn * yn), up(self.Sxs, xn * s),
                                                             up(self.Sss, s * s), up(self.Ssy, s * yn))
        if not self.bias:
            self.g_hat = ((self.Sxy + self.eps) / (self.Sxx + self.eps)).clamp(*self.g_range)
        else:                                                                # [[Sxx+ε, Sxs],[Sxs, Sss+ε]] [g; b] = [Sxy+ε; Ssy]
            a11, a12, a22 = self.Sxx + self.eps, self.Sxs, self.Sss + self.eps
            r1, r2 = self.Sxy + self.eps, self.Ssy
            det = (a11 * a22 - a12 * a12).clamp_min(1e-12)
            self.g_hat = ((a22 * r1 - a12 * r2) / det).clamp(*self.g_range)
            self.b_hat = ((a11 * r2 - a12 * r1) / det).clamp(-self.b_range, self.b_range)
        self.x_f = torch.where(fresh, 0.0, x_new)
        self.w_prev = w_n

    def _thrusts(self, o, state, env):
        self._learn(state)
        s = o["f_acc"] / G
        fed = (o["alpha_cmd"] - self.b_hat * self.scale * s[:, None]) / self.g_hat if self.bias else o["alpha_cmd"] / self.g_hat
        f_acc = o["f_acc"] if self.fz is None else o["f_acc"] / self.fz.update(state, env)
        T = super()._thrusts(dict(o, alpha_cmd=fed, f_acc=f_acc), state, env)
        self.fed_prev, self.s_prev, self.sat_prev = fed, s, torch.as_tensor(self._sat, device=fed.device)
        return T


class B3INDI(B1QP):
    """B3：增量非線性動態反轉（INDI）。Smeur et al. 2016（JGCD, adaptive INDI）、Sun et al. 2020（arXiv 2002.07837）。
      Δu = G⁺(α_cmd − α_f)，u = T_f + Δu；α_f = 陀螺儀差分（含雜訊）經低通的角加速度。
      關鍵同步：「馬達狀態」T_f 也是同一個低通的輸出——送出的指令先過標稱馬達一階模型（時間常數用標稱值 env.tau），
        取一個控制週期內的平均，再過和 α_f 同一個一階低通。這樣 α_f 與 T_f 的延遲相同，α_cmd − α_f 才是「還差多少」。
      總推力同理增量化：ΣT = ΣT_f + M(f_cmd − f_f)，f_f = 加速度計（各模組 z 比力平均，含雜訊）的同一個低通，
        ΣT_f 用瞬時馬達狀態經同一個低通（加速度計是瞬時量，所以不用一個週期平均）。f_cmd = f_acc·(b3d·b3)（投影到目前機體 z，同 Lee 的推力式）。
      G = 標稱形狀的 J⁻¹A（只用標稱 J、A）。分配：把 [ΣT_f + M(…), τ(T_f) + J(α_cmd − α_f)] 當目標，
        用 B1-QP 的有界最小平方（ridge 拉向 T_f，所以是最小增量）；不含陀螺項（α_f 的量測已經包含）。
      adapt=True（B3+adapt）：G 的逐軸對角縮放 ĝ（實際 α 響應 = ĝ·標稱 G ΔT）。線上 RLS：
        y = α_f（扣掉陀螺項）、x = 標稱 G·T_f（兩者經同一低通），含截距的中心化迴歸（截距吸收常數力矩偏置），
        Δu = G⁺ diag(1/ĝ)(α_cmd − α_f)。"""
    name = "B3 INDI"

    def __init__(self, gains=None, tau_f=0.03, adapt=False, lam=0.995, eps=0.01, g_range=(0.2, 5.0), seed=0, **kw):
        super().__init__(gains, **kw)
        self.tau_f, self.adapt, self.lam, self.eps, self.g_range, self.seed = tau_f, adapt, lam, eps, g_range, seed
        self.name = "B3+adapt" if adapt else "B3 INDI"

    def _setup(self, env):
        super()._setup(env)
        B, R4, tt = env.B, 4 * self.N, self.tt
        self.dt = env.cfg.dt * env.cfg.substeps
        self.c = 1 - float(np.exp(-self.dt / self.tau_f))
        self.beta = 1 - (1 - env.cfg.dt / env.tau) ** env.cfg.substeps
        self.hover = self.M * G / R4
        self.At = self._tensor(env.nominal["A"])
        self.Jinv = self._tensor(np.linalg.inv(self.J))
        self.Gn = self.Jinv @ self.At[1:]                                  # [3, 4N]：標稱 G（T → α）
        self.scale = self._tensor([self._amax(a, env) for a in range(3)])
        z = lambda *s: torch.zeros(B, *s, **tt)
        h = torch.full((B, R4), self.hover, **tt)
        self.T_prev, self.T_m, self.T_f = h.clone(), h.clone(), h.clone()
        self.Ts_f, self.f_f = h.sum(1), torch.full((B,), G, **tt)
        self.a_f, self.gy_f, self.w_prev = z(3), z(3), z(3)
        self.g_hat = torch.ones(B, 3, **tt)
        self.S1, self.Sx, self.Sy, self.Sxx, self.Sxy = z(3), z(3), z(3), z(3), z(3)
        self.gen = torch.Generator(device=env.device).manual_seed(self.seed)

    def _thrusts(self, o, state, env):
        tt, c = self.tt, self.c
        B = state["w"].shape[0]
        fresh = (state["t"] == 0)
        fr = fresh[:, None]
        rn = lambda *s: torch.randn(*s, generator=self.gen, **tt)
        w_n = state["w"] + env.cfg.gyro_noise * rn(B, 3)
        f_meas = env.accelerometer()[..., 2].mean(1) + env.cfg.accel_noise * rn(B)
        hov = torch.full_like(self.T_m, self.hover)
        # --- 馬達模型（標稱 τ）、量測與濾波 ---
        T_m_new = torch.where(fr, hov, self.T_m + self.beta * (self.T_prev - self.T_m))
        T_avg = torch.where(fr, hov, 0.5 * (self.T_m + T_m_new))
        w_mid = 0.5 * (w_n + self.w_prev)
        alpha_meas = (w_n - self.w_prev) / self.dt
        gyro = torch.cross(w_mid, w_mid @ self.Jt.T, dim=-1) @ self.Jinv.T
        self.a_f = torch.where(fr, 0.0, self.a_f + c * (alpha_meas - self.a_f))
        self.gy_f = torch.where(fr, 0.0, self.gy_f + c * (gyro - self.gy_f))
        self.T_f = torch.where(fr, hov, self.T_f + c * (T_avg - self.T_f))
        self.Ts_f = torch.where(fresh, hov.sum(1), self.Ts_f + c * (T_m_new.sum(1) - self.Ts_f))
        self.f_f = torch.where(fresh, f_meas, self.f_f + c * (f_meas - self.f_f))
        self.T_m, self.w_prev = T_m_new, w_n
        if self.adapt: self._learn(fresh)
        # --- 增量指令 ---
        R = quat_to_rot(state["q"])
        f_cmd = o["f_acc"] * (o["R_d"][:, :, 2] * R[:, :, 2]).sum(-1).clamp_min(0.0)
        d_alpha = (o["alpha_cmd"] - self.a_f) / self.g_hat
        wr = torch.cat([(self.Ts_f + self.M * (f_cmd - self.f_f))[:, None],
                        self.T_f @ self.At[1:].T + d_alpha @ self.Jt.T], -1)
        T = self._solve(wr, self.T_f / self.tmax)
        self.T_prev = T.clamp(0, self.tmax)
        return T

    def _learn(self, fresh):
        """中心化 RLS：y = ĝ·x + 截距。x = 標稱 G·T_f，y = α_f 扣陀螺項；都除以 α_max。"""
        x = (self.T_f @ self.Gn.T) / self.scale
        y = (self.a_f - self.gy_f) / self.scale
        up = lambda S, v: torch.where(fresh[:, None], 0.0, self.lam * S + v)
        self.S1, self.Sx, self.Sy = up(self.S1, 1.0), up(self.Sx, x), up(self.Sy, y)
        self.Sxx, self.Sxy = up(self.Sxx, x * x), up(self.Sxy, x * y)
        cxx = self.Sxx - self.Sx ** 2 / self.S1.clamp_min(1e-9)
        cxy = self.Sxy - self.Sx * self.Sy / self.S1.clamp_min(1e-9)
        self.g_hat = ((cxy + self.eps) / (cxx + self.eps)).clamp(*self.g_range)


class B1QPInformedYawLim(YawLimMixin, B1QPInformed): pass


class B1QPEstim(B1QP):
    """B1-QP-estim（G2 估計式 informed 對手）：B1-QP 的分配，但質量、慣量、CoM 位置由線上估計取代標稱值。
    只用控制器本來就有的量：陀螺儀（含 env 的雜訊）、加速度計（含雜訊）、自己送出的指令經標稱一階馬達模型。
    (a) 總質量：f_meas（各模組 z 比力平均）= ΣT / M，過原點的純量 RLS（帶遺忘）：y = κ x，κ̂ = M_nom/M̂。
        x = 「指令驅動」的 ΣT_驅動/(M_nom g)（標稱馬達模型，初值 0）；y = (f_meas − f0 (1−β)^k)/g，f0 = episode 開頭量到的比力
        （= ΣT0/M），扣掉未知開頭推力的衰減項，所以不必假設開頭推力是標稱 hover，也不必等馬達模型對齊。
        可選的修正 corr：比力平均含 (α × d̄)_z，d̄ = 模組中心平均相對真實 CoM = −ĉ，用 α̂（陀螺差分）與目前的 ĉ 扣掉。
    (b) CoM 偏移 ĉ_xy：真實力矩 τ = τ_c(T) − ĉ × (ΣT ẑ)（τ_c = 標稱 A 對標稱質心的力矩）。殘差 J α + ω×Jω − τ_c = (−ĉ_y, ĉ_x)·ΣT，
        對 ΣT 做每軸純量 RLS（J 用目前的估計模型；α = 陀螺差分；T 用標稱馬達模型的一個週期平均）。
    (c) 模型重建：payload 當質點，m̂_p = M̂ − M_nom，位置 p̂ = ĉ M̂ / m̂_p（限制在構型外圍內；snap=True 時吸附到最近的模組中心），
        用 p̂ 重算 CoM、J（平行軸）與轉子相對 CoM 的 A，和 B1QPInformed 同一套公式。每個環境各自一份矩陣，
        無界解用 Woodbury 閉式（4x4），超界的環境才用 lsq_linear。
    保護：前 warm_c / warm_m 步分別用標稱 ĉ / M̂；M̂ 限制在 [M_nom, m_max·M_nom]；|ĉ| ≤ c_frac·(構型最大半徑)；
    m̂_p < 1% M_nom 時不重建 payload 質點；每個 episode 開頭重置估計。外迴路 α 限幅仍用標稱（同 informed）。
    無 payload 時 M̂→標稱、ĉ→0，行為≈B1-QP。"""
    name = "B1-QP-estim"

    def __init__(self, gains=None, warm_c=3, warm_m=10, lam_m=0.99, eps_m=0.1, lam_c=0.995, eps_c=0.05, m_max=1.6,
                 c_frac=0.4, snap=False, corr=True, seed=0, **kw):
        super().__init__(gains, **kw)
        self.warm_c, self.warm_m, self.c_frac = warm_c, warm_m, c_frac
        self.lam_m, self.eps_m, self.lam_c, self.eps_c = lam_m, eps_m, lam_c, eps_c
        self.m_max, self.snap, self.corr, self.seed = m_max, snap, corr, seed

    def _setup(self, env):
        super()._setup(env)
        N, B, td = self.N, env.B, torch.float64
        self.td = td
        self.dev = env.device
        mod = self.mod
        self.c0 = torch.as_tensor(np.asarray(mod["d"], float), dtype=td, device=self.dev)            # [N,3] 模組中心相對標稱質心
        self.A0 = torch.as_tensor(np.asarray(mod["A"], float), dtype=td, device=self.dev)            # [4,4N]
        self.J0t = torch.as_tensor(self.J0, dtype=td, device=self.dev)
        self.m0, self.Mn = float(env.p["mass"]), float(env.p["mass"]) * N
        self.r_max = float(self.c0[:, :2].norm(dim=-1).max())
        self.mp_min = 0.01 * self.Mn
        self.c_max = self.c_frac * self.r_max
        self.dt = env.cfg.dt * env.cfg.substeps
        self.beta = 1 - (1 - env.cfg.dt / env.tau) ** env.cfg.substeps
        self.hover_n = self.Mn * G / (4 * N)
        z = lambda *s: torch.zeros(B, *s, dtype=td, device=self.dev)
        self.T_m = torch.full((B, 4 * N), self.hover_n, dtype=td, device=self.dev)
        self.T_prev, self.w_prev = self.T_m.clone(), z(3)
        self.Td, self.f0, self.decay = z(), z(), torch.ones(B, dtype=td, device=self.dev)
        self.Sxy, self.Sxx, self.Sss, self.Sux, self.Suy = z(), z(), z(), z(), z()
        self.kappa, self.c_raw = torch.ones(B, dtype=td, device=self.dev), z(2)
        self.gen = torch.Generator(device=self.dev).manual_seed(self.seed)
        self.M_hat = torch.full((B,), self.Mn, dtype=td, device=self.dev)
        self.c_used = z(2)
        self.Jh, self.A_b = self._model(self.M_hat, z(2))

    def _model(self, Mh, ch):
        """(M̂ [B], ĉ [B,2]) → (J [B,3,3], A [B,4,4N])，並更新 self.c_used（實際用的 ĉ）。
        A 用 ĉ（限制在 c_max 內）；m̂_p > 1% M_nom 時才把 payload 質點（p̂ = ĉ M̂/m̂_p）併進 J。"""
        N, td = self.N, self.td
        mp = (Mh - self.Mn).clamp_min(0.0)
        ok = mp > self.mp_min
        cn = ch.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        ch = ch * (self.c_max / cn).clamp(max=1.0)
        p = ch * (Mh / mp.clamp_min(1e-9))[:, None]
        pn = p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        p = p * (self.r_max / pn).clamp(max=1.0)
        if self.snap:
            idx = (p[:, None, :] - self.c0[None, :, :2]).norm(dim=-1).argmin(1)
            p = torch.where(ok[:, None], self.c0[idx, :2], p)
            ch = torch.where(ok[:, None], mp[:, None] * p / Mh[:, None], ch)
        mp = torch.where(ok, mp, torch.zeros_like(mp))
        self.c_used = ch
        cu3 = torch.cat([ch, torch.zeros_like(ch[:, :1])], -1)
        p3 = torch.cat([p, torch.zeros_like(p[:, :1])], -1)
        eye = torch.eye(3, dtype=td, device=self.dev)
        P = lambda d: (d * d).sum(-1)[..., None, None] * eye - d[..., :, None] * d[..., None, :]
        d = self.c0[None] - cu3[:, None]                                                         # [B,N,3]
        J = N * self.J0t + self.m0 * P(d).sum(1) + mp[:, None, None] * P(p3 - cu3)
        A = self.A0.expand(len(Mh), -1, -1).clone()
        A[:, 1] = self.A0[1] - ch[:, 1:2]
        A[:, 2] = self.A0[2] + ch[:, 0:1]
        return J, A

    def _estimate(self, state, env):
        td, B = self.td, state["w"].shape[0]
        fresh = state["t"] == 0
        fr = fresh[:, None]
        rn = lambda *s: torch.randn(*s, generator=self.gen, dtype=td, device=self.dev)
        w_n = state["w"].double() + env.cfg.gyro_noise * rn(B, 3)
        f_meas = env.accelerometer()[..., 2].double().mean(1) + env.cfg.accel_noise * rn(B)
        hov = torch.full_like(self.T_m, self.hover_n)
        T_new = torch.where(fr, hov, self.T_m + self.beta * (self.T_prev - self.T_m))
        T_avg = torch.where(fr, hov, 0.5 * (self.T_m + T_new))
        self.T_m = T_new
        alpha = (w_n - self.w_prev) / self.dt
        w_mid = 0.5 * (w_n + self.w_prev)
        self.w_prev = w_n
        use = (~fresh).to(td)
        zero = lambda S, v: torch.where(fresh, torch.zeros_like(S), v)
        # (a) 質量
        y = f_meas
        if self.corr:
            y = y - (-alpha[:, 0] * self.c_used[:, 1] + alpha[:, 1] * self.c_used[:, 0])
        self.f0 = torch.where(fresh, f_meas, self.f0)
        self.decay = torch.where(fresh, torch.ones_like(self.decay), self.decay * (1 - self.beta))
        self.Td = torch.where(fresh, torch.zeros_like(self.Td), self.Td + self.beta * (self.T_prev.sum(1) - self.Td))
        x = self.Td / (self.Mn * G)
        y = y - self.f0 * self.decay
        self.Sxy = zero(self.Sxy, self.lam_m * self.Sxy + use * x * y / G)
        self.Sxx = zero(self.Sxx, self.lam_m * self.Sxx + use * x * x)
        self.kappa = ((self.Sxy + self.eps_m) / (self.Sxx + self.eps_m)).clamp(1.0 / self.m_max, 1.0)
        # (b) CoM 偏移
        Jh = self.Jh
        Ja = torch.einsum("bij,bj->bi", Jh, alpha)
        gy = torch.cross(w_mid, torch.einsum("bij,bj->bi", Jh, w_mid), dim=-1)
        r = (Ja + gy)[:, :2] - T_avg @ self.A0[1:3].T
        s = T_avg.sum(1) / (self.Mn * G)
        u = r / (self.Mn * G)
        self.Sss = zero(self.Sss, self.lam_c * self.Sss + use * s * s)
        self.Sux = zero(self.Sux, self.lam_c * self.Sux + use * u[:, 0] * s)
        self.Suy = zero(self.Suy, self.lam_c * self.Suy + use * u[:, 1] * s)
        den = self.Sss + self.eps_c
        self.c_raw = torch.stack([self.Suy / den, -self.Sux / den], -1)                           # (ĉ_x, ĉ_y)
        # 保護：warm 步內用標稱
        Mh = torch.where(state["t"] >= self.warm_m, self.Mn / self.kappa, torch.full_like(self.kappa, self.Mn))
        ch = torch.where((state["t"] >= self.warm_c)[:, None], self.c_raw, torch.zeros_like(self.c_raw))
        self.M_hat = Mh
        self.Jh, self.A_b = self._model(Mh, ch)

    def _thrusts(self, o, state, env):
        self._estimate(state, env)
        td = self.td
        w = state["w"].double()
        al = o["alpha_cmd"].double()
        tau = torch.einsum("bij,bj->bi", self.Jh, al) + torch.cross(w, torch.einsum("bij,bj->bi", self.Jh, w), dim=-1)
        wr = torch.cat([(self.M_hat * o["f_acc"].double())[:, None], tau], -1)
        T = self._solve_b(wr, wr[:, 0:1] / (4 * self.N * self.tmax))
        self.T_prev = T.clamp(0, self.tmax).to(td)
        return T

    def _solve_b(self, wr, x_ref):
        """每個環境各自的 A：min ‖S(A T − wr)‖² + ridge‖x − x_ref‖²，0 ≤ x ≤ 1。無界解 = Woodbury 閉式。"""
        td, N, tmax = self.td, self.N, self.tmax
        W = torch.as_tensor(self.weights, dtype=td, device=self.dev)
        cap = self.A_b.abs().sum(2) * tmax / torch.tensor([1., 2., 2., 2.], dtype=td, device=self.dev)
        S = W / cap                                                                             # [B,4]
        Ah = S[:, :, None] * self.A_b * tmax                                                    # [B,4,4N]
        r = wr * S
        xr = x_ref.expand(-1, 4 * N)
        K = Ah @ Ah.transpose(1, 2) + self.ridge * torch.eye(4, dtype=td, device=self.dev)
        x = xr + torch.einsum("bji,bj->bi", Ah, torch.linalg.solve(K, r - torch.einsum("bij,bj->bi", Ah, xr)))
        x = x.cpu().numpy()
        bad = np.where(((x < -1e-9) | (x > 1 + 1e-9)).any(1))[0]
        Ahn, rn, xrn = Ah.cpu().numpy(), r.cpu().numpy(), xr.cpu().numpy()
        for b in bad:
            Mat = np.vstack([Ahn[b], np.sqrt(self.ridge) * np.eye(4 * N)])
            rhs = np.r_[rn[b], np.sqrt(self.ridge) * xrn[b]]
            x[b] = lsq_linear(Mat, rhs, bounds=(0.0, 1.0), method="bvls").x
        self.n_qp += len(bad)
        self._sat = np.zeros(len(x), bool); self._sat[bad] = True
        return self._tensor(x * tmax)


class B1QPEstimYawLim(YawLimMixin, B1QPEstim): pass


# ============================================================================
# 5) B5 天真
# ============================================================================
class B5(Classical):
    """每個模組都以為自己是單機：用自己 4 顆馬達的 pinv，wrench = [m f_acc, J0 α_cmd + ω×J0 ω]。
    外迴路（f_acc、α_cmd）仍用整架的量測，和其他方法一樣由 leader 算好廣播；天真的只有分配層。"""
    name = "B5"

    def _setup(self, env):
        A1 = assemble([(0, 0)], env.p, env.pitch)["A"]                             # 單機的 A [4, 4]
        self.pinvA1 = self._tensor(np.linalg.pinv(A1))
        self.J0t = self._tensor(self.J0)

    def _thrusts(self, o, state, env):
        wr = torch.cat([(self.m * o["f_acc"])[:, None], self._wrench_torque(o["alpha_cmd"], state["w"], self.J0t)], -1)
        return (wr @ self.pinvA1.T).repeat(1, self.N)                              # 每個模組同一份


# ============================================================================
# 6) 工廠
# ============================================================================
METHOD_NAMES = ["B1", "B1-QP", "B2 k=0", "B2 k=1", "B2 k=2",
                "B2+adapt k=0", "B2+adapt k=1", "B2+adapt k=2", "B5"]


def make_controller(name, gains=None, **kw):
    if name == "B1": return B1(gains)
    if name == "B1-QP": return B1QP(gains, **kw)
    if name == "B5": return B5(gains)
    if name == "B1-QP-informed": return B1QPInformed(gains, **kw)
    if name == "B1-QP+adapt": return B1QPAdapt(gains, **kw)
    if name == "B3 INDI": return B3INDI(gains, **kw)
    if name == "B3+adapt": return B3INDI(gains, adapt=True, **kw)
    if name == "B4": return B1QPAdapt(gains, bias=True, **kw)
    if name == "B1-QP-yawlim": return B1QPYawLim(gains, **kw)
    if name == "B1-QP-informed-yawlim": return B1QPInformedYawLim(gains, **kw)
    if name == "B1-QP-estim": return B1QPEstim(gains, **kw)
    if name == "B1-QP-estim-yawlim": return B1QPEstimYawLim(gains, **kw)
    import re
    mm = re.fullmatch(r"B2-QP-coord-(v\d)(\+adapt)?(-yawlim)? k=(\d)", name)         # 例：B2-QP-coord-v3+adapt-yawlim k=2
    if mm:
        v, ad, yl, k = mm.groups(); k = int(k); kw = {**B2QPCoord.VARIANTS[v], **kw}
        cls = {(0, 0): B2QPCoord, (1, 0): B2QPCoordAdapt, (1, 1): B2QPCoordYawLim}[(bool(ad), bool(yl))]
        o = cls(k, gains, **kw); o.name = name; return o
    if name.startswith("B2-QP+adapt k="): return B2QPAdapt(int(name[-1]), gains, **kw)
    if name.startswith("B2-QP k="): return B2QP(int(name[-1]), gains, **kw)
    if name.startswith("B2+adapt k="): return B2Adapt(int(name[-1]), gains, **kw)
    if name.startswith("B2 k="): return B2(int(name[-1]), gains)
    raise ValueError(name)


# ============================================================================
# 7) P3d 2^3 因子實驗用的組合（QP × yawlim × estim；不改既有類別）
# ============================================================================
class NoQPAllocMixin:
    """把 B1QPEstim 的有界最小平方分配換成 B1 式的無界分配：T = A_b⁺ w（每個環境各自的估計 A_b 的偽逆），
    之後由 Classical.act 直接 clip 到 [0, Tmax]（和 B1 相同；沒有 QP 的飽和處理）。估計部分完全不變。"""

    def _solve_b(self, wr, x_ref):
        A = self.A_b                                                                  # [B,4,4N]
        K = A @ A.transpose(1, 2)
        T = torch.einsum("bji,bj->bi", A, torch.linalg.solve(K, wr.unsqueeze(-1)).squeeze(-1))
        self._sat = np.zeros(len(T), bool)
        return self._tensor(T.cpu().numpy())


class B1YawLim(YawLimMixin, B1): pass                                    # 無 QP + yawlim
class B1Estim(NoQPAllocMixin, B1QPEstim): pass                           # 無 QP + estim
class B1EstimYawLim(YawLimMixin, B1Estim): pass                          # 無 QP + estim + yawlim

P3D_CLASSES = {"B1-yawlim": B1YawLim, "B1-estim": B1Estim, "B1-estim-yawlim": B1EstimYawLim}


def make_controller_p3d(name, gains=None, **kw):
    """make_controller 加上 P3d 新增的三個組合；其餘轉給 make_controller。"""
    if name in P3D_CLASSES: return P3D_CLASSES[name](gains, **kw)
    return make_controller(name, gains, **kw)


# ============================================================================
# 8) P3d 偏離中心掛點：真正知道 payload 偏移的特權 oracle（不改既有類別）
# ============================================================================
def build_with_offset(cells, payload, host, offset, p, pitch):
    """同 analysis.joint_loads.build，但 payload 質點掛在模組 host 的中心 + offset（m）。回傳 dict 與 build 相同。
    payload 的轉動慣量以平行軸定理併入 J；轉子位置 pos = 模組中心相對 CoM + 轉子局部位置（A 的第 2、3 列只用 x、y）。"""
    mod = build_joint_model(cells, p=p, pitch=pitch)                  # 無 payload 的標稱模型（取 c、MOTOR 排列）
    c, m, N = mod["c"], p["mass"], mod["N"]
    pp = c[host] + np.asarray(offset, float)
    mass_m = np.full(N, m)
    M = mass_m.sum() + payload
    com = ((mass_m[:, None] * c).sum(0) + payload * pp) / M
    d = c - com
    dp = pp - com
    par = lambda mi, di: mi * (di @ di * np.eye(3) - np.outer(di, di))
    J = sum(mod["J0"] + par(m, di) for di in d) + par(payload, dp)
    pos = np.vstack([di + np.r_[xy, 0] for di in d for xy in MOTOR_XY])
    A = np.vstack([np.ones(4 * N), pos[:, 1], -pos[:, 0], p["km_ratio"] * mod["spin"]])
    return dict(mod, mass=mass_m, M=M, com=com, d=d, J=J, pos=pos, A=A)


class B1QPInformedOffset(B1QPInformed):
    """B1-QP-informed，且知道 payload 的掛點偏移（payload_offset，相對掛載模組中心）。其餘同 B1QPInformed。
    payload_offset = (0,0,0) 時與 B1QPInformed 的分配模型相同。"""
    name = "B1-QP-informed-offset"

    def __init__(self, gains=None, payload_offset=(0., 0., 0.), **kw):
        super().__init__(gains, **kw)
        self.payload_offset = tuple(payload_offset)

    def _setup(self, env):
        N, m = env.N, env.p["mass"]
        mp = self.payload_frac / (1 - self.payload_frac) * N * m if self.payload_frac > 0 else 0.0
        self.true = build_with_offset(env.cells, mp, self.payload_host % N, self.payload_offset, env.p, env.pitch)
        self.M, self.J = self.true["M"], self.true["J"]
        B1QP._setup(self, env)


class B1QPInformedOffsetYawLim(YawLimMixin, B1QPInformedOffset): pass


def make_controller_p3d_off(name, gains=None, **kw):
    """make_controller_p3d 再加 informed-offset 兩個名稱。"""
    if name == "B1-QP-informed-offset": return B1QPInformedOffset(gains, **kw)
    if name == "B1-QP-informed-offset-yawlim": return B1QPInformedOffsetYawLim(gains, **kw)
    return make_controller_p3d(name, gains, **kw)


# ============================================================================
# 9) RQ-D：中途 payload 事件用的控制器（不改既有類別）
# ============================================================================
class B1QPEstimReset(B1QPEstim):
    """B1-QP-estim + 殘差突變偵測 + 重置。env 的 payload 在事件時刻突然改變，原本的估計器（遺忘 0.99／0.995 的 RLS，
    質量通道還以 episode 開頭的 f0 為錨）要很久才收斂。這個版本每步算兩個「事前殘差」（用上一步的估計預測這一步）：
      質量通道 e_m = f_meas/g − κ̂·ΣT_模型/(M_nom g)（f_meas = 加速度計比力平均，ΣT_模型 = 標稱馬達模型追蹤的推力和；
        馬達模型是精確的，所以不必知道開頭推力；扣掉和父類別相同的 (α×ĉ)_z 修正項）；
      重心通道 e_cx = u_x + ĉ_y s、e_cy = u_y − ĉ_x s（u = 力矩殘差/(M_nom g)，s = ΣT/(M_nom g)；父類別 RLS 的預測式）。
    每個殘差除以自己的滑動標準差（EMA，更新時把 e² 夾在 (3σ)² 以內，事件本身不會把 σ 撐大太快），做 CUSUM：
    S ← max(0, S + |e|/σ − k)；任一通道 S > h 就判定突變（前 warm_det 步和重置後 refr 步不偵測）。
    判定後：該環境的 RLS 統計量（Sxy、Sxx、Sss、Sux、Suy）清零，κ̂ = 1、ĉ = 0，CUSUM 歸零；
    質量通道改成「直接模式」：x = ΣT_模型/(M_nom g)、y = f_meas/g，過原點 RLS（不再需要 f0 錨，因為 hover 時 x ≈ 1，幾步就收斂）。
    沒有突變時行為與 B1QPEstim 完全相同。超參數 k、h、chan（"m"、"c"、"mc" 哪些通道參與偵測）。"""
    name = "B1-QP-estim-reset"

    def __init__(self, gains=None, k=1.0, h=12.0, chan="mc", warm_det=60, refr=30, sig_rate=0.02, **kw):
        super().__init__(gains, **kw)
        self.k, self.h, self.chan, self.warm_det, self.refr, self.sig_rate = k, h, chan, warm_det, refr, sig_rate
        self.name = "B1-QP-estim-reset"

    def _setup(self, env):
        super()._setup(env)
        B, td, dev = env.B, self.td, self.dev
        z = lambda: torch.zeros(B, dtype=td, device=dev)
        self.cs = torch.zeros(B, 3, dtype=td, device=dev)                 # CUSUM：m、cx、cy
        self.var = torch.full((B, 3), 1e-6, dtype=td, device=dev)
        self.direct = torch.zeros(B, dtype=torch.bool, device=dev)
        self.hold = torch.zeros(B, dtype=torch.long, device=dev)
        self.n_alarm = torch.zeros(B, dtype=torch.long, device=dev)       # 診斷：每個環境累計警報次數
        self.alarm_t = torch.full((B,), -1, dtype=torch.long, device=dev) # 診斷：最近一次警報的 episode 內步數
        self.first_alarm_t = torch.full((B,), -1, dtype=torch.long, device=dev)   # 診斷：本 episode 第一次警報
        self.f_prev = z()

    def _estimate(self, state, env):
        td, B = self.td, state["w"].shape[0]
        t = state["t"]
        fresh = t == 0
        fr = fresh[:, None]
        rn = lambda *s: torch.randn(*s, generator=self.gen, dtype=td, device=self.dev)
        w_n = state["w"].double() + env.cfg.gyro_noise * rn(B, 3)
        f_meas = env.accelerometer()[..., 2].double().mean(1) + env.cfg.accel_noise * rn(B)
        hov = torch.full_like(self.T_m, self.hover_n)
        T_new = torch.where(fr, hov, self.T_m + self.beta * (self.T_prev - self.T_m))
        T_avg = torch.where(fr, hov, 0.5 * (self.T_m + T_new))
        self.T_m = T_new
        alpha = (w_n - self.w_prev) / self.dt
        w_mid = 0.5 * (w_n + self.w_prev)
        self.w_prev = w_n
        use = (~fresh).to(td)
        zero = lambda S, v: torch.where(fresh, torch.zeros_like(S), v)
        # 新 episode：偵測相關狀態重置
        self.cs = torch.where(fr, torch.zeros_like(self.cs), self.cs)
        self.var = torch.where(fr, torch.full_like(self.var, 1e-6), self.var)
        self.direct = torch.where(fresh, torch.zeros_like(self.direct), self.direct)
        self.hold = torch.where(fresh, torch.zeros_like(self.hold), self.hold)
        self.first_alarm_t = torch.where(fresh, torch.full_like(self.first_alarm_t, -1), self.first_alarm_t)
        corr = (-alpha[:, 0] * self.c_used[:, 1] + alpha[:, 1] * self.c_used[:, 0]) if self.corr else torch.zeros_like(f_meas)
        sumT = T_new.sum(1)
        # ---- 事前殘差 ----
        Jh = self.Jh
        Ja = torch.einsum("bij,bj->bi", Jh, alpha)
        gy = torch.cross(w_mid, torch.einsum("bij,bj->bi", Jh, w_mid), dim=-1)
        r = (Ja + gy)[:, :2] - T_avg @ self.A0[1:3].T
        s = T_avg.sum(1) / (self.Mn * G)
        u = r / (self.Mn * G)
        e_m = (f_meas - corr) / G - self.kappa * sumT / (self.Mn * G)
        e_cx = u[:, 0] + self.c_raw[:, 1] * s
        e_cy = u[:, 1] - self.c_raw[:, 0] * s
        e = torch.stack([e_m, e_cx, e_cy], -1)
        z_ = e.abs() / self.var.sqrt()
        active = (t >= self.warm_det) & (self.hold <= 0)
        cs_new = (self.cs + z_ - self.k).clamp_min(0.0)
        mask = torch.tensor([("m" in self.chan), ("c" in self.chan), ("c" in self.chan)], device=self.dev)
        cs_new = torch.where(mask[None], cs_new, torch.zeros_like(cs_new))
        self.cs = torch.where(active[:, None], cs_new, self.cs)
        alarm = active & (cs_new.amax(1) > self.h)
        # σ 更新（事件前才學；夾住大殘差）
        n = t.clamp_min(1).to(td)[:, None]
        rate = torch.maximum(1.0 / n, torch.full_like(n, self.sig_rate))
        e2 = torch.minimum(e * e, 9 * self.var)
        e2 = torch.where(t[:, None] < 3, e * e, e2)
        self.var = torch.where(fr, self.var, (self.var + rate * (e2 - self.var)).clamp_min(1e-12))
        self.hold = self.hold - 1
        # ---- 重置 ----
        self.n_alarm += alarm.long()
        self.alarm_t = torch.where(alarm, t, self.alarm_t)
        self.first_alarm_t = torch.where(alarm & (self.first_alarm_t < 0), t, self.first_alarm_t)
        self.direct = self.direct | alarm
        self.cs = torch.where(alarm[:, None], torch.zeros_like(self.cs), self.cs)
        self.hold = torch.where(alarm, torch.full_like(self.hold, self.refr), self.hold)
        for nm in ("Sxy", "Sxx", "Sss", "Sux", "Suy"):
            S = getattr(self, nm); setattr(self, nm, torch.where(alarm, torch.zeros_like(S), S))
        self.kappa = torch.where(alarm, torch.ones_like(self.kappa), self.kappa)
        self.c_raw = torch.where(alarm[:, None], torch.zeros_like(self.c_raw), self.c_raw)
        # ---- (a) 質量：一般模式（父類別，以 f0 為錨）或直接模式 ----
        self.f0 = torch.where(fresh, f_meas, self.f0)
        self.decay = torch.where(fresh, torch.ones_like(self.decay), self.decay * (1 - self.beta))
        self.Td = torch.where(fresh, torch.zeros_like(self.Td), self.Td + self.beta * (self.T_prev.sum(1) - self.Td))
        x_a = self.Td / (self.Mn * G)
        y_a = (f_meas - corr) - self.f0 * self.decay
        x_d = sumT / (self.Mn * G)
        y_d = (f_meas - corr)
        x = torch.where(self.direct, x_d, x_a)
        y = torch.where(self.direct, y_d, y_a)
        self.Sxy = zero(self.Sxy, self.lam_m * self.Sxy + use * x * y / G)
        self.Sxx = zero(self.Sxx, self.lam_m * self.Sxx + use * x * x)
        self.kappa = ((self.Sxy + self.eps_m) / (self.Sxx + self.eps_m)).clamp(1.0 / self.m_max, 1.0)
        # ---- (b) CoM ----
        self.Sss = zero(self.Sss, self.lam_c * self.Sss + use * s * s)
        self.Sux = zero(self.Sux, self.lam_c * self.Sux + use * u[:, 0] * s)
        self.Suy = zero(self.Suy, self.lam_c * self.Suy + use * u[:, 1] * s)
        den = self.Sss + self.eps_c
        self.c_raw = torch.stack([self.Suy / den, -self.Sux / den], -1)
        Mh = torch.where(t >= self.warm_m, self.Mn / self.kappa, torch.full_like(self.kappa, self.Mn))
        ch = torch.where((t >= self.warm_c)[:, None], self.c_raw, torch.zeros_like(self.c_raw))
        self.M_hat = Mh
        self.Jh, self.A_b = self._model(Mh, ch)


class B1QPEstimResetYawLim(YawLimMixin, B1QPEstimReset): pass


class B1QPInformedEvent(B1QPEstim):
    """RQ-D 上限：知道事件的一切（時刻、掛點模組、質量比例、偏移）的特權 oracle。每步依 env.ev_mode / t_event / ev_host
    用真實的模型（build_with_offset：質量、J、轉子相對 CoM 的 A）取代估計：事件前後各用對的那一份，事件那一步（t == t_event）起切到事件後。
    其餘（分配、外迴路）與 B1QPEstim 相同。不是可部署的控制器，只當上限。frac、offset 要與 env 的 payload_event 一致。"""
    name = "B1-QP-informed-event"

    def __init__(self, gains=None, frac=0.15, offset=None, **kw):
        super().__init__(gains, **kw)
        self.frac, self.offset = frac, tuple(offset) if offset is not None else (0., 0., 0.)

    def _setup(self, env):
        super()._setup(env)
        N, td, dev, m = self.N, self.td, self.dev, env.p["mass"]
        mp = self.frac / (1 - self.frac) * N * m
        models = [build_with_offset(env.cells, mp, h, self.offset, env.p, env.pitch) for h in range(N)]
        nom = build_with_offset(env.cells, 0.0, 0, (0., 0., 0.), env.p, env.pitch)
        T = lambda x: torch.as_tensor(np.asarray(x, float), dtype=td, device=dev)
        self.J_h, self.A_h, self.M_h = T([mo["J"] for mo in models]), T([mo["A"] for mo in models]), T([mo["M"] for mo in models])
        self.J_n, self.A_n, self.M_n = T(nom["J"]), T(nom["A"]), float(nom["M"])

    def _estimate(self, state, env):
        t, h, mode = state["t"], env.ev_host, env.ev_mode
        present = torch.where(mode == 1, t >= env.t_event, torch.where(mode == 2, t < env.t_event, torch.zeros_like(t, dtype=torch.bool)))
        p3 = present[:, None, None]
        self.Jh = torch.where(p3, self.J_h[h], self.J_n.expand(len(t), 3, 3))
        self.A_b = torch.where(p3, self.A_h[h], self.A_n.expand(len(t), 4, 4 * self.N))
        self.M_hat = torch.where(present, self.M_h[h], torch.full_like(self.M_h[h], self.M_n))
        self.c_used = torch.zeros(len(t), 2, dtype=self.td, device=self.dev)


class B1QPInformedEventYawLim(YawLimMixin, B1QPInformedEvent): pass


def make_controller_rqd(name, gains=None, **kw):
    """make_controller_p3d_off 再加 RQ-D 的三個控制器名稱。"""
    if name == "B1-QP-estim-reset": return B1QPEstimReset(gains, **kw)
    if name == "B1-QP-estim-reset-yawlim": return B1QPEstimResetYawLim(gains, **kw)
    if name == "B1-QP-informed-event": return B1QPInformedEvent(gains, **kw)
    if name == "B1-QP-informed-event-yawlim": return B1QPInformedEventYawLim(gains, **kw)
    return make_controller_p3d_off(name, gains, **kw)
