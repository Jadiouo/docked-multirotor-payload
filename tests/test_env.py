"""P2 環境驗收 1–4（docs/env_spec.md）＋ 形狀、reset、auto-reset 的基本測試。物理測試一律用 float64。"""
import numpy as np
import pytest
import torch
from quadsim.assembly import CONFIGS, assemble
from quadsim.env import ModularEnv, EnvConfig, Perturb, TASKS
from quadsim.module_params import load_module
from quadsim import mjcf
from analysis.joint_loads import build, module_loads, joints, project, edges

P, PITCH, LIMITS = load_module()
G = 9.81
F64 = EnvConfig(dtype=torch.float64)


def make(cells, B=1, task="T2", cfg=F64, **kw):
    return ModularEnv(CONFIGS[cells], B, task, seed=0, cfg=cfg, **kw)


def hover_thrust(env):
    """用 A 的 pinv 解出 hover 推力（合力 = Mg，三軸力矩 = 0）。"""
    A = env.nominal["A"]
    return np.linalg.pinv(A) @ np.array([env.nominal["M"] * G, 0, 0, 0])


def quat_angle(q1, q2):
    """兩姿態的夾角（rad）。用相對四元數的向量部分算，避免 arccos 在 0 附近的精度損失。"""
    w1, v1 = q1[0], q1[1:]; w2, v2 = q2[0], q2[1:]
    rel_w = w1 * w2 + v1 @ v2                              # q1 ⊗ conj(q2)
    rel_v = w2 * v1 - w1 * v2 - np.cross(v1, v2)
    return 2 * np.arctan2(np.linalg.norm(rel_v), abs(rel_w))


# ---------------------------------------------------------------- 驗收 1：和 MuJoCo 比對
@pytest.mark.parametrize("name", ["line3", "L3", "grid6"])
def test_matches_mujoco(name, capsys):
    rng = np.random.default_rng(1)
    env = make(name)
    asm = env.nominal
    model, data = mjcf.make_reference(asm, asm["c"])
    assert model.opt.timestep == env.cfg.dt == 0.002
    # 非平凡的初始狀態：有速度、有角速度、隨意姿態
    pos, vel, omega = np.array([0.1, -0.2, 1.0]), rng.normal(size=3), rng.normal(size=3) * 2
    quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
    env.set_state(pos=pos[None], vel=vel[None], q=quat[None], w=omega[None])
    mjcf.set_state(model, data, pos, vel, quat, omega)
    max_p = max_q = moved = rotated = 0.0
    for k in range(500):                                   # 1 s；推力每 5 個子步（一個控制步）換一次
        if k % 5 == 0: T = rng.uniform(0, env.tmax, 4 * asm["N"])
        env._advance(torch.as_tensor(T[None]))
        mjcf.step(model, data, asm, T)
        max_p = max(max_p, np.abs(env.pos[0].numpy() - data.qpos[:3]).max())
        max_q = max(max_q, quat_angle(env.q[0].numpy(), data.qpos[3:7]))
        moved = max(moved, np.linalg.norm(data.qpos[:3] - pos)); rotated = max(rotated, quat_angle(quat, data.qpos[3:7]))
    with capsys.disabled():
        print(f"\n[MuJoCo 比對] {name}: 最大位置誤差 {max_p:.3e} m，最大姿態誤差 {max_q:.3e} rad"
              f"（軌跡位移 {moved:.2f} m、轉動 {rotated:.2f} rad）")
    assert moved > 0.1 and rotated > 0.3                   # 軌跡要夠有料，比對才有意義
    assert max_p < 1e-3 and max_q < 1e-3


# ---------------------------------------------------------------- 驗收 1b：獨立的力模型檢查
def geometric_reference(env, payload_frac, dt=0.002):
    """MuJoCo 側：body 原點 = 幾何原點，inertial 在 CoM；M、CoM、J 在這裡用 numpy 獨立算（含 payload 質點，掛在最後一個模組）。"""
    asm, m, N = env.nominal, P["mass"], env.N
    c = np.c_[np.asarray(env.cells, float) * PITCH, np.zeros(N)]                 # 模組中心（幾何原點座標）
    mp = payload_frac / (1 - payload_frac) * N * m if payload_frac else 0.0
    masses, pts = [m] * N + [mp], list(c) + [c[-1]]
    M = sum(masses); com = sum(mi * pi for mi, pi in zip(masses, pts)) / M
    J0 = np.diag([P["Ixx"], P["Iyy"], P["Izz"]])
    J = sum(J0 for _ in range(N)) + sum(mi * ((pi - com) @ (pi - com) * np.eye(3) - np.outer(pi - com, pi - com)) for mi, pi in zip(masses, pts))
    rotor_geo = asm["rotor_pos"] + asm["com"]                                    # assemble 的轉子位置是相對標稱 CoM，換回幾何原點
    model, data = mjcf.make_reference(dict(M=M, J=J), c, dt=dt, inertial_pos=com)
    return model, data, com, rotor_geo, P["km_ratio"] * asm["rotor_spin"]


@pytest.mark.parametrize("name,payload", [("line3", 0.3), ("L3", 0.0)])
def test_matches_mujoco_independent_force_model(name, payload, capsys):
    """body 原點不在 CoM 時，MuJoCo 積分的是原點、我們積分的是 CoM，兩者的半隱式 Euler 離散化差 O(dt)（原點繞 CoM 轉動的二階項），
    不是力模型的錯。所以這個測試用 dt = 1e-4 s 降低離散化差，1 s = 10000 子步；推力仍每 0.01 s 換一次。"""
    dt, hold = 1e-4, 100
    rng = np.random.default_rng(2)
    env = make(name, perturb=Perturb(payload_frac=payload) if payload else None, cfg=EnvConfig(dtype=torch.float64, dt=dt))
    model, data, com, rotor_geo, kspin = geometric_reference(env, payload, dt)
    assert np.allclose(env.geo["M"][0].item(), model.body_mass[1])
    pos, vel, omega = np.array([0.1, -0.2, 1.0]), rng.normal(size=3), rng.normal(size=3) * 2
    quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
    env.set_state(pos=pos[None], vel=vel[None], q=quat[None], w=omega[None])      # env 的 pos 是 CoM
    mjcf.set_state(model, data, pos, vel, quat, omega)                            # 先放好姿態，再把 body 原點換算成 CoM 在 pos
    R0 = data.xmat[1].reshape(3, 3)
    data.qpos[:3] = pos - R0 @ com
    data.qvel[:3] = vel - np.cross(R0 @ omega, R0 @ com)                          # 原點速度 = CoM 速度 − ω×(R com)
    import mujoco; mujoco.mj_forward(model, data)
    max_p = max_q = 0.0
    for k in range(int(1.0 / dt)):
        if k % hold == 0: T = rng.uniform(0, env.tmax, 4 * env.N)
        env._advance(torch.as_tensor(T[None]))
        mjcf.step_per_rotor(model, data, rotor_geo, kspin, T)
        max_p = max(max_p, np.abs(env.pos[0].numpy() - data.xipos[1]).max())      # xipos = 世界座標 CoM
        max_q = max(max_q, quat_angle(env.q[0].numpy(), data.qpos[3:7]))
    with capsys.disabled():
        print(f"\n[MuJoCo 獨立力模型] {name} payload={payload}: 最大位置誤差 {max_p:.3e} m，最大姿態誤差 {max_q:.3e} rad")
    assert max_p < 1e-3 and max_q < 1e-3


def test_mujoco_pitch_direction_physical():
    """只加大 +x 端轉子推力 → 機頭抬起，τy < 0、繞 y 的角速度 < 0（由 MuJoCo 判斷，不用我們的公式）。"""
    env = make("line3")
    model, data, com, rotor_geo, kspin = geometric_reference(env, 0.0)
    T = hover_thrust(env).copy()
    T[rotor_geo[:, 0] > rotor_geo[:, 0].max() - 0.05] += 0.03                    # 最右邊那個模組的 4 顆
    mjcf.set_state(model, data, np.array([0, 0, 1.0]) - com, np.zeros(3), np.array([1., 0, 0, 0]), np.zeros(3))
    for _ in range(50): mjcf.step_per_rotor(model, data, rotor_geo, kspin, T)
    wb, Rm = data.qvel[3:6], data.xmat[1].reshape(3, 3)
    assert wb[1] < 0 and abs(wb[1]) > 5 * max(abs(wb[0]), 1e-12), wb          # τy < 0：繞 y 負向轉
    assert Rm[2, 0] > 0                                                          # 機頭（body x 軸）抬起：R 的 x 軸 z 分量為正
    print("\n[pitch] 機體 ω =", wb, " R[2,0] =", Rm[2, 0])


# ---------------------------------------------------------------- 驗收 2：hover 平衡
@pytest.mark.parametrize("name", ["single", "line3", "grid6"])
def test_hover_drift(name):
    env = make(name)
    T = hover_thrust(env)
    env.set_state(pos=env.hover_goal[None], thrust=T[None])
    u = torch.as_tensor(2 * T / env.tmax - 1).reshape(1, env.N, 4)
    p0 = env.pos.clone()
    for k in range(env.cfg.episode_steps - 1):             # 499 步 + 最後一步（truncated 後 reset）= 5 s
        env.step(u)
        assert (env.pos - p0).abs().max() < 1e-6
    _, _, term, trunc, info = env.step(u)
    assert trunc.all() and not term.any()
    assert (info["final_state"]["pos"] - p0).abs().max() < 1e-6


# ---------------------------------------------------------------- 驗收 3：加速度計
@pytest.mark.parametrize("name", ["single", "line3", "grid6"])
def test_accelerometer_hover_and_yaw_spin(name):
    env = make(name)
    T = hover_thrust(env)
    env.set_state(thrust=T[None])
    acc = env.accelerometer()[0].numpy()
    assert np.allclose(acc, [0, 0, G], atol=1e-9)             # 靜止 hover：每個模組都是 +g ẑ
    W = 3.0                                                   # 純 yaw 旋轉：α = 0，剩向心項 ω×(ω×d) = −W²(dx, dy, 0)
    env.set_state(w=[[0, 0, W]])
    d = np.asarray(env.nominal["c"])                          # 模組中心相對 CoM（手算）
    expect = np.c_[-W ** 2 * d[:, :2], np.full(len(d), G)]
    assert np.allclose(env.accelerometer()[0].numpy(), expect, atol=1e-9)
    assert np.allclose(env._alpha.numpy(), 0, atol=1e-9)
    # 同樣的量也要出現在觀測裡（關掉雜訊）
    obs = env.observe(noise=False)[0].numpy()
    assert np.allclose(obs[:, 15:18], expect, atol=1e-9)


# ---------------------------------------------------------------- 驗收 4：接頭內力 = joint_loads 的瞬間結果
@pytest.mark.parametrize("name,payload", [("line3", 0.0), ("T4", 0.0), ("grid6", 0.0), ("line3", 0.3)])
def test_joint_loads_match_analysis(name, payload):
    pert = Perturb(payload_frac=payload, payload_host=-1) if payload else None
    env = make(name, B=5, perturb=pert)
    N, rng = env.N, np.random.default_rng(3)
    T = rng.uniform(0, env.tmax, (5, 4 * N)); w = rng.normal(size=(5, 3)) * 3
    q = rng.normal(size=(5, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)   # 任意姿態：重力在機體系與慣性力抵消，結果應與姿態無關
    env.set_state(thrust=T, w=w, q=q)
    comp = env.joint_components().numpy()
    mp = float(env.geo["mp"][0])
    mod = build(CONFIGS[name], payload=mp, host=N - 1, p=P, pitch=PITCH)
    E = edges(CONFIGS[name])
    for b in range(5):
        rows = project(mod, joints(mod, module_loads(mod, T[b], w[b])))
        for row in rows:                                   # 靜定構型的 dict 順序是自葉往根，所以用 edge 對應
            c = comp[b, E.index(row["edge"])]
            got = dict(axial=abs(c[0]), shear_ip=abs(c[1]), shear_oop=abs(c[2]), torsion=abs(c[3]), bend=np.hypot(c[4], c[5]))
            for k, v in got.items():
                assert abs(v - row[k]) < 1e-9, (name, b, E[e], k, v, row[k])


def test_joint_ratio_uses_limits():
    env = make("line2")
    env.set_state(thrust=np.zeros((1, 8)))                    # 沒推力：全架自由落體，每個模組只受自己的推力 0 → 接頭內力 0
    assert env.joint_load_ratio().item() < 1e-9
    env.set_state(thrust=np.array([[env.tmax] * 4 + [0.0] * 4]))   # 一個模組全推、另一個不推：接頭要傳力
    c = env.joint_components()[0, 0].numpy()
    want = max(abs(c[0]) / LIMITS["pull_N"], np.hypot(c[1], c[2]) / LIMITS["shear_N"], np.hypot(c[4], c[5]) / LIMITS["peel_moment_Nm"])
    assert env.joint_load_ratio().item() == pytest.approx(want) and want > 0.01


# ---------------------------------------------------------------- 形狀、物理誤差、reset
@pytest.mark.parametrize("task", TASKS)
def test_shapes_and_reset(task):
    env = ModularEnv(CONFIGS["L3"], 16, task, seed=1)
    obs = env.reset()
    assert obs.shape == (16, 3, env.obs_dim) == (16, 3, 28) and torch.isfinite(obs).all()
    assert env.adjacency.shape == (3, 3, 4) and env.adjacency.sum() == 4   # L3 有 2 條邊 × 2 個方向
    obs, r, term, trunc, info = env.step(torch.rand(16, 3, 4) * 2 - 1)
    assert obs.shape == (16, 3, 28) and r.shape == term.shape == trunc.shape == (16,)
    assert info["joint_load_ratio"].shape == (16,) and info["final_obs"].shape == obs.shape


def test_neighbor_flags_and_blind():
    env = make("L3")                                        # (0,0),(1,0),(0,1)
    flags = env.observe(noise=False)[0, :, 20:24]
    assert flags.tolist() == [[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    env.cfg.blind = True
    assert env.observe(noise=False)[..., 20:24].abs().sum() == 0


def test_autoreset_same_step_semantics():
    env = ModularEnv(CONFIGS["line2"], 4, "T1", seed=0)
    env.set_state(pos=[[0, 0, -0.01]] * 4)                  # 高度 < 0 → 立刻 terminated
    obs, r, term, trunc, info = env.step(torch.zeros(4, 2, 4))
    assert term.all() and not trunc.any()
    assert (info["final_state"]["pos"][:, 2] < 0).all()      # final 是 reset 前的狀態
    assert (info["state"]["pos"][:, 2] > 0).all() and (env.t == 0).all()   # 目前狀態已 reset
    assert not torch.allclose(info["final_obs"], obs)
    assert (r < -10).all()                                  # 含終止懲罰


def test_truncation_at_500_steps():
    env = make("single", task="T2")
    T = hover_thrust(env); env.set_state(thrust=T[None])
    u = torch.as_tensor(2 * T / env.tmax - 1).reshape(1, 1, 4)
    for k in range(499):
        _, _, term, trunc, _ = env.step(u)
        assert not trunc.any()
    _, _, term, trunc, _ = env.step(u)
    assert trunc.all() and not term.any()


def test_payload_and_dock_perturbation():
    env = make("line3", B=64, perturb=Perturb(payload_frac=0.3, payload_host=2, dock_delta=0.003, dock_tilt=0.05, thrust_eps=0.1))
    g, m = env.geo, P["mass"]
    assert torch.allclose(g["M"], torch.full((64,), 3 * m / 0.7, dtype=torch.float64))   # payload = 30% 總質量
    assert (g["kmax"] / env.tmax).min() >= 0.9 and (g["kmax"] / env.tmax).max() <= 1.1
    # 真實 CoM 相對模組中心 d_i 的質量加權和 + payload 項 = 0
    mp = g["mp"][0]
    assert torch.allclose(m * g["d"].sum(1) + mp * g["dp"], torch.zeros(64, 3, dtype=torch.float64), atol=1e-12)
    assert g["tdir"][..., :2].abs().max() > 0                # 推力方向確實有傾斜
    env.reset(); env.step(torch.zeros(64, 3, 4))             # 可正常跑


def test_deterministic_seed():
    a = ModularEnv(CONFIGS["line3"], 8, "T1", seed=5); b = ModularEnv(CONFIGS["line3"], 8, "T1", seed=5)
    u = torch.rand(8, 3, 4) * 2 - 1
    assert torch.equal(a.reset(), b.reset()) and torch.equal(a.step(u)[0], b.step(u)[0])


def test_t2_goal_sequence_paired_per_episode():
    # 兩個方法在不同時機失敗：同一個 (環境槽, episode 編號) 的 T2 目標序列仍然相同
    def run(kill_at):
        env = ModularEnv(CONFIGS["single"], 4, "T2", seed=3)
        env.set_state(pos=[[0, 0, 1.5]] * 4)
        seqs = {}                                                # (槽, episode 編號, 段) → (目標, yaw)
        for s in range(700):
            for b in range(4):
                seqs.setdefault((b, int(env.ep[b]), int(env.t[b]) // 150), (env.goal_pos[b].clone(), env.goal_yaw[b].clone()))
            if s in kill_at:
                env.set_state(idx=[kill_at[s]], pos=[[0, 0, -1.0]])    # 讓該槽立刻墜地終止
            env.step(torch.zeros(4, 1, 4))
        return seqs
    a, b = run({10: 0, 200: 1}), run({60: 0, 90: 2, 400: 1})
    common = set(a) & set(b)
    assert any(k[1] >= 1 for k in common)                       # 確實比到了第二個以後的 episode
    for k in common:
        assert torch.equal(a[k][0], b[k][0]) and torch.equal(a[k][1], b[k][1]), k
    g = torch.stack([a[k][0] for k in sorted(a) if k[2] == 0])   # 不同 (槽, episode) 的目標確實不同、在範圍內
    assert g.unique(dim=0).shape[0] == g.shape[0]
    assert ((g[:, :2].abs() <= 1.0).all())
