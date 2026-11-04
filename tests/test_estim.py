"""估計式 informed 古典對手（B1QPEstim）的測試。執行：.venv/bin/python -m pytest -q tests/test_estim.py"""
import numpy as np
import torch
from quadsim.env import ModularEnv, Perturb
from quadsim.assembly import CONFIGS
from quadsim.controllers import B1QP, B1QPEstim, B1QPInformed, goal_of, make_controller
from analysis.joint_loads import build

torch.set_num_threads(1)


def hover_run(cells, pt, ctrl, steps=300, n=4):
    env = ModularEnv(cells, n, task="T1", seed=0, perturb=pt)
    env.set_state(pos=env.hover_goal, vel=0., w=0., q=[1., 0, 0, 0], goal_pos=env.hover_goal, goal_yaw=0.)
    st = env.state()
    for _ in range(steps):
        _, _, te, _, info = env.step(ctrl.act(st, goal_of(st), env))
        assert not te.any()
        st = info["state"]
    return env, st


def true_com_shift(env):
    return -env.geo["d"][0].mean(0)[:2].numpy()           # 真實 CoM 相對標稱質心（d = 模組中心 − 真實 CoM）


def test_no_payload_estimates_nominal_and_matches_b1qp():
    cells = CONFIGS["line6"]
    ce, cq = B1QPEstim(), B1QP()
    env, st = hover_run(cells, None, ce)
    _, stq = hover_run(cells, None, cq)
    assert abs(float(ce.M_hat.mean()) / ce.Mn - 1) < 0.03
    assert float(ce.c_raw.abs().max()) < 0.003
    assert float(ce.c_used.abs().max()) < 0.003
    assert (st["pos"] - stq["pos"]).norm(dim=-1).max() < 0.02                 # 靜態 hover：位置與 B1-QP 幾乎相同


def test_static_hover_converges_to_true_payload():
    for name, frac, host in [("line6", 0.15, 5), ("line8", 0.15, 7), ("grid10", 0.12, 9)]:
        cells = CONFIGS[name]
        ctrl = B1QPEstim()
        env, _ = hover_run(cells, Perturb(payload_frac=frac, payload_host=host), ctrl)
        Mt = float(env.geo["M"][0])
        assert abs(float(ctrl.M_hat.mean()) / Mt - 1) < 0.10, name
        ct = true_com_shift(env)
        ch = ctrl.c_raw.mean(0).numpy()
        assert np.linalg.norm(ch - ct) < 0.10 * np.linalg.norm(ct), (name, ch, ct)


def test_rebuilt_model_equals_informed_when_estimates_exact():
    cells = CONFIGS["grid10"]
    host, frac = 9, 0.15
    ctrl = B1QPEstim()
    env, _ = hover_run(cells, Perturb(payload_frac=frac, payload_host=host), ctrl, steps=2, n=2)
    N = len(cells)
    mp = frac / (1 - frac) * N * env.p["mass"]
    tr = build(cells, payload=mp, host=host, p=env.p, pitch=env.pitch)
    nom = build(cells, p=env.p, pitch=env.pitch)
    ch = torch.tensor((tr["com"] - nom["com"])[:2], dtype=torch.float64)[None]
    J, A = ctrl._model(torch.tensor([tr["M"]], dtype=torch.float64), ch)
    assert np.allclose(J[0].numpy(), tr["J"], rtol=1e-6, atol=1e-9)
    assert np.allclose(A[0].numpy(), tr["A"], rtol=1e-6, atol=1e-9)
    # 估計為標稱時，重建 = 標稱模型
    J0, A0 = ctrl._model(torch.tensor([nom["M"]], dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64))
    assert np.allclose(J0[0].numpy(), nom["J"], rtol=1e-6) and np.allclose(A0[0].numpy(), nom["A"], rtol=1e-6)


def test_batched_solve_equals_b1qp_solve_nominal():
    cells = CONFIGS["line3"]
    ce, cq = B1QPEstim(), B1QP()
    env = ModularEnv(cells, 3, task="T1", seed=0)
    st = env.state()
    ce.act(st, goal_of(st), env); cq.act(st, goal_of(st), env)
    ce.Jh, ce.A_b = ce._model(torch.full((3,), ce.Mn, dtype=torch.float64), torch.zeros(3, 2, dtype=torch.float64))
    wr = torch.tensor([[1.0 + 0.1 * i, 0.002, -0.001, 0.0005] for i in range(3)], dtype=torch.float64)
    xr = wr[:, 0:1] / (4 * ce.N * ce.tmax)
    a, b = ce._solve_b(wr, xr), cq._solve(wr.float(), xr.float())
    assert torch.allclose(a, b, atol=1e-4)


def test_factory_names():
    assert make_controller("B1-QP-estim").name == "B1-QP-estim"
    assert make_controller("B1-QP-estim-yawlim").name == "B1-QP-estim-yawlim"
    assert make_controller("B1-QP-informed-yawlim", payload_frac=0.1, payload_host=2).name == "B1-QP-informed-yawlim"
