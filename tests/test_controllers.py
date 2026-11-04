"""古典控制器測試。執行：.venv/bin/python -m pytest -q tests/test_controllers.py"""
import numpy as np
import torch
from quadsim.env import ModularEnv, quat_from_euler
from quadsim.assembly import CONFIGS
from quadsim.controllers import B1, B1QP, B2, B2Adapt, B2QP, B2QPAdapt, B5, Gains, goal_of


def hover_env(name, n=4, task="T1"):
    """水平、靜止、在目標位置的 env（T1 的 reset 是隨機的，所以用 set_state 覆蓋）。"""
    env = ModularEnv(CONFIGS[name], n, task=task, seed=0)
    env.set_state(pos=env.hover_goal, vel=0., w=0., q=[1., 0, 0, 0], goal_pos=env.hover_goal, goal_yaw=0.)
    return env


def tilted_state(env, roll=0.15, pitch=-0.1, w=(0.3, -0.2, 0.0)):
    """小角度傾斜加一點角速度：外迴路會給出中等大小的 α_cmd，不會碰到推力上下限。"""
    q = quat_from_euler(torch.tensor(roll), torch.tensor(pitch), torch.tensor(0.))
    env.set_state(q=q, w=torch.tensor(w))
    return env.state()


def run(env, ctrl, steps):
    st, terminated = env.state(), False
    for _ in range(steps):
        _, _, te, _, info = env.step(ctrl.act(st, goal_of(st), env))
        terminated |= bool(te.any())
        st = info["state"]
    return st, terminated


def test_b1_hover_stable_single_and_line3():
    for name in ["single", "line3"]:
        env = hover_env(name)
        st, terminated = run(env, B1(), 499)                     # 5 s（第 500 步 env 會自動 reset，所以停在 499）
        assert not terminated
        assert (st["pos"] - env.hover_goal).norm(dim=-1).max() < 0.01, name
        assert st["w"].abs().max() < 0.05, name


def test_b1_recovers_from_tilt():
    env = hover_env("line3")
    tilted_state(env, roll=0.5, pitch=-0.4, w=(0., 0., 0.))
    st, terminated = run(env, B1(), 499)
    assert not terminated
    assert (st["pos"] - env.hover_goal).norm(dim=-1).max() < 0.05


def test_b2_full_ball_equals_b1_when_unsaturated():
    # k = 直徑：每個模組的球是整架，分配 = oracle（不含陀螺項，所以這裡 ω 很小）
    for name in ["line3", "L3", "grid6"]:
        env = hover_env(name)
        st = tilted_state(env, w=(0.01, -0.01, 0.0))
        diam = len(CONFIGS[name])                                # 直徑 ≤ N，球半徑 k+1 ≥ N 一定涵蓋整架
        a1 = B1().act(st, goal_of(st), env)
        a2 = B2(diam).act(st, goal_of(st), env)
        assert ((a1 + 1) / 2).min() > 0 and ((a1 + 1) / 2).max() < 1          # 確實沒有飽和
        # 差在陀螺項 ω×Jω（~1e-4 量級），其他完全相同
        assert (a1 - a2).abs().max() < 2e-3, name


def test_b5_equals_b1_on_single():
    env = hover_env("single")
    st = tilted_state(env, w=(1.0, -2.0, 0.5))                   # 含陀螺項，B5 的單機 wrench 也有
    a1, a5 = B1().act(st, goal_of(st), env), B5().act(st, goal_of(st), env)
    assert torch.allclose(a1, a5, atol=1e-5)


def test_b5_worse_than_b1_on_line3():
    # B5 每個模組都當自己是單機：pitch 的轉動慣量（沿 x 排開）比 3 個單機大很多，推力不夠，姿態恢復比 B1 慢。
    # （roll 不行：line3 沿 x 排開，Ixx = 3 個單機的 Ixx，B5 在 roll 上剛好是對的）
    err = {}
    for name, c in [("B1", B1()), ("B5", B5())]:
        env = hover_env("line3")
        tilted_state(env, roll=0., pitch=0.5, w=(0., 0., 0.))
        st, _ = run(env, c, 30)
        err[name] = c.attitude_error(st, goal_of(st)).mean().item()
    assert err["B5"] > 2 * err["B1"]


def test_qp_equals_b1_when_unsaturated_and_respects_bounds_when_saturated():
    env = hover_env("line3")
    st = tilted_state(env, w=(0.01, -0.01, 0.0))
    qp = B1QP()
    assert torch.allclose(qp.act(st, goal_of(st), env), B1().act(st, goal_of(st), env), atol=1e-4)
    assert qp.n_qp == 0                                          # 沒有飽和，沒有呼叫 lsq_linear
    st = tilted_state(env, roll=1.2, pitch=0.)                   # 大角度加很高的姿態增益：一定飽和
    qp = B1QP(Gains(kR=3000.0, kw=100.0))
    a = qp.act(st, goal_of(st), env)
    assert qp.n_qp > 0 and a.min() >= -1 - 1e-6 and a.max() <= 1 + 1e-6


def test_qp_joint_penalty_lowers_joint_load():
    env = hover_env("line8")
    st = tilted_state(env, roll=0.0, pitch=0.3, w=(0., 0., 0.))
    load = {}
    for w_j in [0.0, 1.0]:
        env.set_state(thrust=(B1QP(w_joint=w_j).act(st, goal_of(st), env) + 1).reshape(env.B, -1) / 2 * env.tmax)
        load[w_j] = env.joint_load_ratio().max().item()
    assert load[1.0] <= load[0.0] + 1e-9


def test_adapt_ghat_stays_near_one_when_gain_is_correct():
    # k = N：B2 等於 oracle，分配的增益本來就是 1，ĝ 應該留在 1 附近
    env = ModularEnv(CONFIGS["line3"], 8, task="ATT_STEP", seed=0)
    ctrl = B2Adapt(3)
    st, hist = env.state(), []
    for _ in range(300):
        _, _, te, _, info = env.step(ctrl.act(st, goal_of(st), env))
        st = info["state"]
        hist.append(ctrl.g_hat.clone())
    g = torch.stack(hist)[30:]                                   # 跳過最前面的暫態
    assert (g.median(0).values - 1).abs().max() < 0.15           # 每個環境、每個軸的中位數都在 1 附近
    assert g.min() > 0.5 and g.max() < 2.0


def test_adapt_finds_wrong_gain():
    # k=0 的 line8 增益明顯不是 1（P0）：ĝ 要離開 1，而且在 clip 範圍內
    env = ModularEnv(CONFIGS["line8"], 4, task="ATT_STEP", seed=0)
    ctrl = B2Adapt(0)
    st = env.state()
    for _ in range(150):
        _, _, _, _, info = env.step(ctrl.act(st, goal_of(st), env))
        st = info["state"]
    assert ctrl.g_hat.min() >= 0.05 and ctrl.g_hat.max() <= 20
    assert (ctrl.g_hat[:, :2].log().abs() > 0.1).any()


def test_b2qp_equals_b2_when_unsaturated_and_respects_bounds_when_saturated():
    env = hover_env("line6")
    st = tilted_state(env, w=(0.01, -0.01, 0.0))
    for k in (0, 1, 2):
        qp = B2QP(k)
        a = qp.act(st, goal_of(st), env)
        assert torch.allclose(a, B2(k).act(st, goal_of(st), env), atol=1e-4), k
        assert qp.n_qp == 0
    st = tilted_state(env, roll=1.2, pitch=0.)                   # 大角度加很高的姿態增益：一定飽和
    qp = B2QP(1, Gains(kR=3000.0, kw=100.0))
    a = qp.act(st, goal_of(st), env)
    assert qp.n_qp > 0 and a.min() >= -1 - 1e-6 and a.max() <= 1 + 1e-6 and qp._sat.all()


def test_b2qp_adapt_runs_and_ghat_near_one_for_full_ball():
    env = ModularEnv(CONFIGS["line3"], 4, task="ATT_STEP", seed=0)
    ctrl = B2QPAdapt(3)
    assert ctrl.name == "B2-QP+adapt k=3"
    st = env.state()
    for _ in range(150):
        _, _, te, _, info = env.step(ctrl.act(st, goal_of(st), env))
        st = info["state"]
    assert (ctrl.g_hat - 1).abs().max() < 0.5


# ---------------------------------------------------------------- G2：B3 INDI、B1-QP+adapt、B4
from quadsim.env import Perturb
from quadsim.controllers import B3INDI, B1QPAdapt, make_controller


def test_indi_hover_stable_single_and_line3():
    for name in ["single", "line3"]:
        for adapt in (False, True):
            env = hover_env(name)
            st, terminated = run(env, B3INDI(adapt=adapt), 499)
            assert not terminated
            assert (st["pos"] - env.hover_goal).norm(dim=-1).max() < 0.02, (name, adapt)
            assert st["w"].abs().max() < 0.1, (name, adapt)


def test_indi_unsaturated_step_matches_b1_direction_and_respects_bounds():
    env = hover_env("line3")
    st = tilted_state(env, w=(0.01, -0.01, 0.0))
    c = B3INDI()
    a = c.act(st, goal_of(st), env)
    assert a.min() >= -1 - 1e-6 and a.max() <= 1 + 1e-6
    st = tilted_state(env, roll=1.2, pitch=0.)
    c = B3INDI(Gains(kR=3000.0, kw=100.0))
    a = c.act(st, goal_of(st), env)
    assert a.min() >= -1 - 1e-6 and a.max() <= 1 + 1e-6


def _att_track_err(name, perturb, steps=150, n=16):
    """ATT_STEP（0.8~1.2 rad 的姿態階躍）前 steps 步的平均姿態誤差（rad）。"""
    env = ModularEnv(CONFIGS["line3"], n, task="ATT_STEP", seed=0, perturb=perturb)
    ctrl = make_controller(name)
    st, errs = env.state(), []
    for _ in range(steps):
        _, _, _, _, info = env.step(ctrl.act(st, goal_of(st), env))
        fs = info["final_state"]
        errs.append(ctrl.attitude_error(fs, goal_of(fs)).mean().item())
        st = info["state"]
    return float(np.mean(errs))


def test_indi_beats_b1_attitude_tracking_with_thrust_coefficient_error_on_line3():
    pert = Perturb(thrust_eps=0.2)                              # 每顆馬達推力係數 ±20%（獨立）
    e1, e3 = _att_track_err("B1", pert), _att_track_err("B3 INDI", pert)
    assert e3 < e1, (e1, e3)


def test_b1qp_adapt_and_b4_run_and_b4_finds_com_offset_sign():
    # payload 掛在 line3 的 +x 端：CoM 偏向 +x，標稱分配在 pitch 上有常數偏置；B4 的 b̂ 的 pitch 軸要離開 0
    pert = Perturb(payload_frac=0.1, payload_host=2)
    env = ModularEnv(CONFIGS["line3"], 4, task="POS_STEP", seed=0, perturb=pert)
    ctrl = make_controller("B4")
    st = env.state()
    for _ in range(300):
        _, _, te, _, info = env.step(ctrl.act(st, goal_of(st), env))
        st = info["state"]
    assert ctrl.b_hat[:, 1].abs().max() > 1e-3
    assert not bool(te.all())
    assert isinstance(make_controller("B1-QP+adapt"), B1QPAdapt)


# ---------------------------------------------------------------- B1-QP-informed（G2 特權 oracle）
from quadsim.controllers import B1QPInformed


def test_informed_equals_b1qp_without_payload():
    for name in ["line3", "grid9"]:
        env = hover_env(name)
        st = tilted_state(env)
        a = B1QP().act(st, goal_of(st), env)
        b = B1QPInformed().act(st, goal_of(st), env)
        assert torch.allclose(a, b, atol=1e-6), name
        assert isinstance(make_controller("B1-QP-informed"), B1QPInformed)


def test_informed_matches_true_geometry_with_payload():
    cells = CONFIGS["line6"]
    host = int(np.argmax([c[0] for c in cells]))
    env = ModularEnv(cells, 2, task="T1", seed=0, perturb=Perturb(payload_frac=0.15, payload_host=host))
    c = B1QPInformed(payload_frac=0.15, payload_host=host)
    env.set_state(pos=env.hover_goal, vel=0., w=0., q=[1., 0, 0, 0], goal_pos=env.hover_goal, goal_yaw=0.)
    st = env.state(); c.act(st, goal_of(st), env)
    g = env.geo
    assert abs(c.M - g["M"][0].item()) < 1e-6
    assert np.allclose(c.J, g["J"][0].numpy(), atol=1e-7)
    A_env = np.vstack([g["tdir"][0, :, 2].numpy(), g["tq"][0].numpy().T])
    assert np.allclose(c._A(env), A_env, atol=1e-6)


# ---------------------------------------------------------------- B2-QP-coord（飽和協調的古典法）
from quadsim.controllers import B2QPCoord, B2QPCoordAdapt


def test_coord_equals_b2qp_when_unsaturated():
    env = hover_env("line6")
    st = tilted_state(env, w=(0.01, -0.01, 0.0))
    for k in (1, 2):
        c = B2QPCoord(k)
        a = c.act(st, goal_of(st), env)
        assert torch.allclose(a, B2QP(k).act(st, goal_of(st), env), atol=1e-6), k
        assert c.n_qp == 0 and torch.allclose(c.last_broadcast, torch.ones_like(c.last_broadcast))


def test_coord_broadcast_is_scalars_not_shape_and_acts_when_saturated():
    sizes = set()
    for name in ("line3", "line6", "grid9"):
        env = hover_env(name)
        st = tilted_state(env, roll=1.2, pitch=0.)
        c = B2QPCoord(1, Gains(kR=3000.0, kw=100.0), rounds=2)
        a = c.act(st, goal_of(st), env)
        bc = c.last_broadcast
        sizes.add(tuple(bc.shape[1:]))                              # 每個環境、每回合的廣播向量維度，與構型大小 N 無關
        assert bc.shape == (env.B, 2, 4) and (bc >= 0).all() and (bc <= 1).all()
        assert a.min() >= -1 - 1e-6 and a.max() <= 1 + 1e-6
    assert sizes == {(2, 4)}
    assert c.name == "B2-QP-coord k=1" and B2QPCoordAdapt(2).name == "B2-QP-coord+adapt k=2"
    from quadsim.controllers import make_controller
    for v in ("v1", "v2", "v3"):
        c = make_controller(f"B2-QP-coord-{v}+adapt k=2")
        assert isinstance(c, B2QPCoordAdapt) and c.mode == B2QPCoord.VARIANTS[v]["mode"]
        env = hover_env("line6"); st = tilted_state(env, roll=1.2, pitch=0.)
        c = make_controller(f"B2-QP-coord-{v} k=1", Gains(kR=3000.0, kw=100.0)); c.act(st, goal_of(st), env)
        assert c.last_broadcast.shape[-1] == 4 and c.last_broadcast.shape[1] == B2QPCoord.VARIANTS[v]["rounds"]


def test_yawlim_limits_yaw_ref_rate_and_equals_base_without_yaw_error():
    from quadsim.controllers import make_controller, B1QPYawLim
    env = hover_env("line6", task="T1")
    st = env.state()
    c = make_controller("B1-QP-yawlim", rate_c=1.0)
    assert isinstance(c, B1QPYawLim)
    g = dict(pos=st["goal_pos"], yaw=torch.full((env.B,), 1.5))
    c.act(st, g, env); c.act(st, g, env)
    assert (c.yaw_ref.abs() <= 2 * c.dpsi + 1e-6).all() and (c.yaw_ref > 0).all()
    a = c.act(st, goal_of(st), env)                              # 目標 yaw = 0、目前 yaw = 0：和 B1-QP 一樣
    c2 = make_controller("B1-QP-yawlim"); c2.act(st, goal_of(st), env); c2.yaw_ref.zero_()
    assert torch.allclose(c2.act(st, goal_of(st), env), B1QP().act(st, goal_of(st), env), atol=1e-6)
