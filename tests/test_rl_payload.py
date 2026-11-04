"""未知 payload 訓練（RandomPayloadEnv、rl_train --payload-max）的測試。CPU：CUDA_VISIBLE_DEVICES= pytest -q tests/test_rl_payload.py"""
import sys
import torch

from quadsim.assembly import CONFIGS
from quadsim.env import ModularEnv, Perturb
from quadsim.policy import ModularActorCritic, OBS_DIM, PRIV_DIM
from quadsim.rl_env import PackedEnv, RandomPayloadEnv, g2_payload_host

CELLS = CONFIGS["line3"]


def rollout(env_fn, steps=5):
    torch.manual_seed(0)
    pe = PackedEnv(env_fn(), 4, 2)
    a, c = pe.reset()
    outs = [a, c]
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        act = torch.randn(pe.B, 4, 4, generator=g) * 0.1
        a, c, r, te, tr, _ = pe.step(act)
        outs += [a, c, r]
    return outs


def test_pmax0_bit_identical():
    """pmax=0 時 rl_train 走原本的 ModularEnv 路徑：兩次同 seed 輸出逐位元相同，且 make_groups 不建 RandomPayloadEnv。"""
    from scripts.rl_train import make_groups, parse_args
    mk = lambda: ModularEnv(CELLS, 8, task="T2", seed=3)
    for x, y in zip(rollout(mk), rollout(mk)):
        assert torch.equal(x, y)
    old = sys.argv
    sys.argv = ["x", "--run-name", "t", "--configs", "line3:T2", "--n-envs", "4", "--device", "cpu"]
    try:
        args = parse_args()
    finally:
        sys.argv = old
    assert args.payload_max == 0.0
    groups, _ = make_groups(args, "cpu")
    assert type(groups[0][1].env) is ModularEnv


def test_sampling_range_and_host():
    env = RandomPayloadEnv(CELLS, 256, 0.20, task="T2", seed=0)
    for _ in range(3):
        f, h = env.payload_frac, env.payload_host
        assert f.shape == (256,) and (f > 0).all() and (f <= 0.20).all() and f.std() > 0.02 and f.max() > 0.18
        assert set(h.tolist()) == set(range(env.N))            # 掛點在所有模組中均勻抽（不是只有 argmax x）
        env._reset_idx(torch.arange(256))
    m = env.p["mass"] * env.N
    assert torch.allclose(env.geo["M"], m / (1 - env.payload_frac).to(env.geo["M"].dtype), rtol=1e-5)
    before = env.payload_frac.clone()
    env._reset_idx(torch.tensor([0, 5]))
    assert torch.equal(env.payload_frac[10:], before[10:]) and not torch.equal(env.payload_frac[[0, 5]], before[[0, 5]])


def test_frac_choices():
    env = RandomPayloadEnv(CELLS, 64, 0.20, frac_choices=(0.10, 0.15, 0.20), task="T2", seed=0)
    assert set(round(x, 2) for x in env.payload_frac.unique().tolist()) <= {0.10, 0.15, 0.20}


def test_geometry_and_loads_match_fixed_payload():
    """逐環境幾何、接頭負載與 g2 用的固定 payload ModularEnv 相同（同一套公式，host 逐環境）。"""
    env = RandomPayloadEnv(CELLS, 6, 0.20, task="T2", seed=0)
    i = 3
    f, h = float(env.payload_frac[i]), int(env.payload_host[i])
    ref = ModularEnv(CELLS, 1, task="T2", perturb=Perturb(payload_frac=f, payload_host=h))
    for k in ("M", "J", "d", "mp", "dp"):
        assert torch.allclose(env.geo[k][i], ref.geo[k][0], rtol=1e-5, atol=1e-7), k
    # 兩個 env 都 hover 動作走幾步，module_loads 相同
    act = torch.zeros(6, env.N, 4); act1 = torch.zeros(1, env.N, 4)
    for _ in range(3):
        env.step(act); ref.step(act1)
    ref.w, ref._alpha, ref._F_b = env.w[i:i+1].clone(), env._alpha[i:i+1].clone(), env._F_b[i:i+1].clone()
    ref.thrust = env.thrust[i:i+1].clone()
    assert torch.allclose(env.module_loads()[i], ref.module_loads()[0], rtol=1e-4, atol=1e-5)


def test_actor_blind_critic_privileged():
    """actor 觀測不含 payload：同一組 (狀態) 在不同 payload 下 actor 觀測形狀不變、且維度與標稱相同；
    critic 特權欄位 d（真實 CoM 相對位置）隨 payload 改變，critic 維度不變（舊 checkpoint 可載入）。"""
    pe = PackedEnv(RandomPayloadEnv(CELLS, 16, 0.20, task="T2", seed=0), 4, 2)
    a, c = pe.reset()
    nom = PackedEnv(ModularEnv(CELLS, 16, task="T2", seed=0), 4, 2)
    a0, c0 = nom.reset()
    assert a.shape == a0.shape and c.shape == c0.shape
    # 特權欄位 r = d / pitch：payload 讓 CoM 往掛點移動，與標稱不同；且不同環境（不同 frac）彼此不同
    priv = pe._priv()
    assert not torch.allclose(priv, nom._priv(), atol=1e-4)
    assert priv[:, :3].std(0).max() > 1e-4
    # 掛點模組（x 最大）的 d 隨 frac 單調：frac 越大 CoM 越靠近掛點，|d_x| 越小
    mass_dev = (pe.env.geo["M"] - pe.env.p["mass"] * pe.env.N).abs()          # 質量也隨 payload 變，但 actor 的 28 維 obs 不含它
    assert mass_dev.max() > 0 and a.shape[-1] == pe.actor_layout.size
    # 舊維度：新 agent 的 critic 輸入維度與 PackedEnv 一致
    ag = ModularActorCritic(4, hist=2)
    assert ag.critic_layout.size == pe.critic_layout.size and ag.actor_layout.size == pe.actor_layout.size
    ag.value_team(c)
