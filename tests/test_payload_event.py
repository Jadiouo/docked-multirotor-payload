"""中途 payload 事件（env.Perturb.payload_event、RandomPayloadEnv 的 offset_max／event_prob）與 RQ-D 控制器的測試。
執行：CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q tests/test_payload_event.py"""
import numpy as np
import torch

from quadsim.assembly import CONFIGS
from quadsim.controllers import B1QP, B1QPEstim, B1QPEstimReset, B1QPInformedEvent, goal_of
from quadsim.env import ModularEnv, EnvConfig, Perturb, quat_to_rot
from quadsim.rl_env import RandomPayloadEnv

torch.set_num_threads(1)
LINE3 = CONFIGS["line3"]
HOVER = EnvConfig(pos_step=0.0)


def ev(typ, host="argmax", offset=None, frac=0.15, t_range=(1.0, 3.5)):
    return Perturb(payload_event=dict(type=typ, t_range=t_range, frac=frac, host=host, offset=offset))


def hover_env(cells, n, pt, seed=0):
    return ModularEnv(cells, n, task="POS_STEP", seed=seed, cfg=HOVER, perturb=pt)


def run(env, ctrl, steps):
    st, errs, zs = env.state(), [], []
    for _ in range(steps):
        _, _, te, _, info = env.step(ctrl.act(st, goal_of(st), env))
        assert not te.any()
        st = info["state"]
        errs.append((st["goal_pos"] - st["pos"]).norm(dim=-1)); zs.append(st["pos"][:, 2].clone())
    return torch.stack(errs), torch.stack(zs)


def test_off_is_bit_identical():
    """payload_event=None（預設）：輸出與 seed 相同的另一個 env 逐位元相同，且完全沒啟用事件機制。"""
    def roll(pt):
        env = ModularEnv(LINE3, 8, task="T2", seed=3, perturb=pt)
        assert not env._events and env.geo_alt is None
        g = torch.Generator().manual_seed(1)
        outs = [env.observe(noise=False)]
        for _ in range(30):
            o, r, te, tr, info = env.step(torch.randn(8, 3, 4, generator=g) * 0.1)
            outs += [o, r, info["state"]["pos"], info["state"]["q"]]
        return outs
    for a, b in zip(roll(None), roll(Perturb(payload_event=None))):
        assert torch.equal(a, b)


def test_attach_mass_and_event_time():
    env = hover_env(LINE3, 4096, ev("attach", host="random"))
    m0 = env.p["mass"] * env.N
    t = env.t_event
    assert (t >= 100).all() and (t <= 350).all()
    assert abs(float(t.float().mean()) - 225) < 5 and t.min() < 110 and t.max() > 340           # U(1.0, 3.5) s → 步 100..350 大致均勻
    assert torch.allclose(env.geo["M"], torch.full_like(env.geo["M"], m0))                         # 事件前無 payload
    assert torch.allclose(env.geo_alt["M"], torch.full_like(env.geo["M"], m0 / (1 - 0.15)), rtol=1e-5)
    assert set(env.ev_host.tolist()) == {0, 1, 2}                                                  # random 掛點
    # 實際跑：事件步前後總質量 M0 → M0/(1-f)
    env = hover_env(LINE3, 4, ev("attach")); ctrl = B1QP(); st = env.state()
    te0 = env.t_event.clone()
    for s in range(int(te0.max()) + 2):
        before = env.geo["M"].clone()
        _, _, te, _, info = env.step(ctrl.act(st, goal_of(st), env)); st = info["state"]
        for b in range(4):
            if s == int(te0[b]):
                assert torch.isclose(before[b], torch.tensor(m0), rtol=1e-5)
                assert torch.isclose(env.geo["M"][b], torch.tensor(m0 / 0.85), rtol=1e-5)


def test_drop_reverses_and_velocity_field_continuous():
    for typ in ("attach", "drop"):
        env = hover_env(LINE3, 4, ev(typ, host="random", offset=(0.02, -0.02, -0.01)))
        m0 = env.p["mass"] * env.N
        assert torch.allclose(env.geo["M"], torch.full((4,), m0 / 0.85 if typ == "drop" else m0), rtol=1e-5)
        # 給定線速度、角速度與姿態，直接觸發事件：模組 0 中心的世界位置、世界速度要連續
        env.set_state(vel=[0.3, -0.2, 0.1], w=[0.5, -0.4, 0.8], q=[0.9, 0.2, -0.3, 0.1])
        env.q /= env.q.norm(dim=-1, keepdim=True)
        env.t[:] = env.t_event
        R = quat_to_rot(env.q)
        c_pos = lambda: env.pos + (R * env.geo["d"][:, 0][:, None]).sum(-1)
        c_vel = lambda: env.vel + torch.cross((R * env.w[:, None]).sum(-1), (R * env.geo["d"][:, 0][:, None]).sum(-1), dim=-1)
        p0, v0, w0 = c_pos(), c_vel(), env.w.clone()
        env._apply_events()
        assert env.ev_done.all() and torch.equal(env.w, w0)
        assert torch.allclose(c_pos(), p0, atol=1e-6) and torch.allclose(c_vel(), v0, atol=1e-6)
        assert torch.allclose(env.geo["M"], torch.full((4,), m0 if typ == "drop" else m0 / 0.85), rtol=1e-5)
        assert not torch.allclose(env.geo["d"], env.geo_alt["d"])                                  # CoM 真的換了


def test_unknown_payload_sinks_informed_switches():
    cells = LINE3
    n = 4
    # 不知道 payload 的 B1-QP：attach 後掉高度；知道事件時刻的 informed 版事件後幾乎沒有誤差
    e1 = hover_env(cells, n, ev("attach"), seed=2); c1 = B1QP()
    err1, z1 = run(e1, c1, 500)
    e2 = hover_env(cells, n, ev("attach"), seed=2); c2 = B1QPInformedEvent(frac=0.15)
    t_ev = e2.t_event.clone()
    err2, z2 = run(e2, c2, 500)
    post = lambda err: torch.stack([err[int(t_ev[b]):int(t_ev[b]) + 100, b].max() for b in range(n)])
    assert (post(err1) > 0.05).all()
    assert (post(err2) < 0.06).all() and (post(err2) < 0.5 * post(err1)).all()
    # 事件前 informed 用標稱、事件後用含 payload 的模型
    e3 = hover_env(cells, 2, ev("attach"), seed=2); c3 = B1QPInformedEvent(frac=0.15)
    st = e3.state(); tev = int(e3.t_event[0])
    for s in range(tev + 2):
        act = c3.act(st, goal_of(st), e3)          # 這一步的 act 用的是 state["t"] == s；env 在 t == t_event 的 step 開頭才換幾何
        if s == tev - 1: assert abs(float(c3.M_hat[0]) - c3.Mn) < 1e-9
        if s == tev: assert abs(float(c3.M_hat[0]) - float(e3.geo_alt["M"][0])) < 1e-6 * c3.Mn
        _, _, _, _, info = e3.step(act)
        st = info["state"]
    assert abs(float(e3.geo["M"][0]) - float(c3.M_hat[0])) < 1e-6 * c3.Mn


def test_reset_estimator_detects_event_and_matches_when_quiet():
    cells = CONFIGS["line6"]
    # 無事件：不報警，行為與 B1QPEstim 相同
    pt = Perturb()
    ea, eb = hover_env(cells, 4, pt), hover_env(cells, 4, pt)
    ca, cb = B1QPEstim(), B1QPEstimReset()
    ea_err, _ = run(ea, ca, 300); eb_err, _ = run(eb, cb, 300)
    assert int(cb.n_alarm.sum()) == 0
    assert torch.allclose(ea_err, eb_err, atol=1e-9)
    # attach：事件後 20 步內警報，事件前沒有
    e = hover_env(cells, 4, ev("attach", host="random"), seed=1); c = B1QPEstimReset()
    t_ev = e.t_event.clone()
    st = e.state(); first = None
    for s in range(int(t_ev.max()) + 40):
        _, _, te, _, info = e.step(c.act(st, goal_of(st), e)); st = info["state"]
        a = c.first_alarm_t
        if s == int(t_ev.min()) - 2: assert (a < 0).all()                                          # 事件前沒有警報
    assert ((c.first_alarm_t >= t_ev) & (c.first_alarm_t <= t_ev + 20)).all(), (c.first_alarm_t, t_ev)


def test_random_payload_env_offset_and_event():
    env = RandomPayloadEnv(LINE3, 2048, 0.20, task="T2", seed=0, offset_max=1.0, event_prob=0.3)
    frac_ev = float((env.ev_mode > 0).float().mean())
    assert abs(frac_ev - 0.3) < 0.04
    att = env.ev_mode == 1
    m0 = env.p["mass"] * env.N
    assert torch.allclose(env.geo["M"][att], torch.full_like(env.geo["M"][att], m0))            # attach：事件前無 payload
    assert (env.geo_alt["M"][att] > m0).all()                                                    # 事件後有
    drop = env.ev_mode == 2
    assert (env.geo["M"][drop] > m0).all() and torch.allclose(env.geo_alt["M"][drop], torch.full_like(env.geo["M"][drop], m0))
    none = env.ev_mode == 0
    assert (env.geo["M"][none] > m0).all()
    assert (env.t_event[none] > 10 ** 6).all() and (env.t_event[~none] >= 100).all() and (env.t_event[~none] <= 350).all()
    # 偏移範圍：用 payload 位置 = 掛點模組中心 + offset；從 dp 與 d 反推 offset
    b = 0.718 * 0.4 * env.pitch
    bz = 0.718 * 0.03
    mp = env.geo["mp"]; pay = ~att & (mp > 0)
    g = env.geo
    # 掛點模組中心 c_h = d_h + com，payload 位置 = com + dp → offset = dp − d_h
    dh = g["d"][torch.arange(2048), env.payload_host]
    o = (g["dp"] - dh)[pay]
    assert o[:, 0].abs().max() <= b + 1e-5 and o[:, 1].abs().max() <= b + 1e-5
    assert o[:, 2].max() <= 1e-6 and o[:, 2].min() >= -bz - 1e-5
    assert o[:, 0].std() > 0.3 * b and o[:, 2].min() < -0.5 * bz
    # 跑起來事件會觸發
    env2 = RandomPayloadEnv(LINE3, 16, 0.20, task="T2", seed=1, offset_max=1.0, event_prob=1.0)
    assert (env2.ev_mode > 0).all()
    ctrl = B1QP(); st = env2.state(); fired = torch.zeros(16, dtype=torch.bool)
    for _ in range(360):
        _, _, te, _, info = env2.step(ctrl.act(st, goal_of(st), env2)); st = info["state"]
        fired |= env2.ev_done
    assert fired.all()


def test_random_payload_defaults_unchanged():
    """offset_max=0、event_prob=0（預設）：不啟用事件，亂數流與舊版相同（兩次同 seed 逐位元相同，且與明確寫 0 相同）。"""
    a = RandomPayloadEnv(LINE3, 16, 0.20, task="T2", seed=5)
    b = RandomPayloadEnv(LINE3, 16, 0.20, task="T2", seed=5, offset_max=0.0, event_prob=0.0)
    assert not a._events and not b._events
    for k in a.geo: assert torch.equal(a.geo[k], b.geo[k])
    assert float(a.geo["dp"].abs().max()) > 0 and float(((a.geo["dp"] - a.geo["d"][torch.arange(16), a.payload_host]).abs().max())) < 1e-6   # 偏移 0
    oa, ob = a.observe(noise=False), b.observe(noise=False)
    assert torch.equal(oa, ob)


def test_rl_train_flags():
    import sys
    from scripts.rl_train import make_groups, parse_args
    old = sys.argv
    sys.argv = ["x", "--run-name", "t", "--configs", "line3:T2", "--n-envs", "4", "--device", "cpu", "--payload-max", "0.2",
                "--payload-offset-max", "1", "--payload-event-prob", "0.3"]
    try:
        args = parse_args()
    finally:
        sys.argv = old
    assert args.payload_offset_max == 1.0 and args.payload_event_prob == 0.3
    groups, _ = make_groups(args, "cpu")
    e = groups[0][1].env
    assert type(e) is RandomPayloadEnv and e.offset_max == 1.0 and e.event_prob == 0.3
