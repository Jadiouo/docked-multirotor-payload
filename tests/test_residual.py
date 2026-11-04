"""殘差 RL（ResidualEnv、rl_train --residual、rl_eval 的殘差評估）的測試。CPU：CUDA_VISIBLE_DEVICES= pytest -q tests/test_residual.py"""
import sys
import numpy as np
import torch

from quadsim.assembly import CONFIGS
from quadsim.env import ModularEnv, Perturb
from quadsim.policy import ModularActorCritic, OBS_DIM
from quadsim.rl_env import PackedEnv, ResidualEnv, make_estim_yawlim_controller

CELLS = CONFIGS["line3"]


class ZeroPolicy:
    """δ ≡ 0 的假 policy，走真的 rl_eval.run_cell_rl 路徑。"""
    def __init__(self, n_max, scale=0.2):
        self.n_max, self.hist, self.residual_scale = n_max, 1, scale

    def act(self, obs, deterministic=False):
        return torch.zeros(*obs.shape[:-1], 4), None


def test_delta_zero_matches_classical():
    """最重要：δ≡0 時，rl_eval 的殘差評估（ResidualEnv + 假 policy）與 p3d_factorial.run_one（estim-yawlim，snap=False，
    即 g1_gate.run_cell）在 E3 偏離中心 payload 下同 seed 的逐 episode 分數相同。"""
    from scripts.rl_eval import run_cell_rl, load_gains, e3_perturb
    from scripts.p3d_factorial import run_one
    gains, n = load_gains(), 6
    for cfg in ("line3", "line6"):
        cells = CONFIGS[cfg]
        for task in ("POS_STEP", "T2"):
            rl = run_cell_rl(ZeroPolicy(len(cells)), cells, task, n, gains, perturb=e3_perturb(cells))
            ref = run_one(cfg, "offcenter", "B1-QP-estim-yawlim", task, n, gains, nosnap=True)
            for k in ("score", "pos_rms", "att_rms", "success"):
                d = np.abs(np.round(np.asarray(rl[k], float), 5) - np.asarray(ref[k], float)).max()     # run_cell 的輸出四捨五入到 5 位
                print(cfg, task, k, "max|diff| =", d)
                assert d < 1e-9, (cfg, task, k, d)


def test_controller_state_reset_on_autoreset():
    """某個環境 auto-reset 時，只有那個環境的控制器狀態被重置（RLS 統計量、yaw_ref），其他環境不受影響。"""
    env = ModularEnv(CELLS, 3, task="T2", seed=1, perturb=Perturb(payload_frac=0.15, payload_host=2))
    re = ResidualEnv(env, make_estim_yawlim_controller(), 0.2)
    re.reset()
    z = torch.zeros(3, 3, 4)
    for _ in range(30): re.step(z)
    c = re.ctrl
    assert (c.Sxx > 0).all() and (c.decay < 1).all() and (c.Td > 0).all()
    yaw_before = c.yaw_ref.clone()
    env.pos[1, 2] = -1.0                                   # 讓環境 1 墜地 → 下一步 terminated → auto-reset
    _, _, te, tr, info = re.step(z)
    assert te.tolist() == [False, True, False]
    # step 結束時已用 post-reset 狀態（t=0）算過一次下一步的 u_base，所以環境 1 的狀態是「重置後第 0 步」的值
    assert info["state"]["t"].tolist()[1] == 0
    assert c.Sxx[1] == 0 and c.Sss[1] == 0 and c.Td[1] == 0 and c.decay[1] == 1
    assert c.Sxx[0] > 0 and c.Sxx[2] > 0 and c.decay[0] < 1 and c.decay[2] < 1
    q = env.q
    yaw1 = torch.atan2(2 * (q[1, 0] * q[1, 3] + q[1, 1] * q[1, 2]), 1 - 2 * (q[1, 2] ** 2 + q[1, 3] ** 2))
    # fresh 時 yaw_ref 先設成目前 yaw，再最多往目標走一個 dpsi
    assert (c.yaw_ref[1] - yaw1).abs() <= c.dpsi + 1e-6
    assert info["final_obs"].shape[-1] == OBS_DIM + 4 and info["u_base"].shape == (3, 3, 4)


def test_residual_shapes_and_packing():
    env = ModularEnv(CELLS, 4, task="T2", seed=2)
    re = ResidualEnv(env, make_estim_yawlim_controller(), 0.2)
    pe = PackedEnv(re, 4, 2)
    assert re.obs_dim == OBS_DIM + 4 and pe.actor_layout.size == pe.actor_layout.size
    a, c = pe.reset()
    agent = ModularActorCritic(4, hist=2, obs_dim=OBS_DIM + 4, hover_action=0.0, head_std=1e-3, log_std_init=-2.0)
    assert a.shape[-1] == agent.actor_layout.size and c.shape[-1] == agent.critic_layout.size
    act, _ = agent.act(a)
    assert act.abs().max() < 0.5                                         # 初始 δ 很小
    a, c, r, te, tr, info = pe.step(act.reshape(4, 4, 4))
    assert info["final_obs"].shape[-1] == agent.actor_layout.size
    # checkpoint 重建（不傳 obs_scale）
    ModularActorCritic(**agent.hp)


def test_residual_flag_off_is_old_path():
    """--residual 關閉：make_groups 不包 ResidualEnv，policy 參數與舊版相同。"""
    from scripts.rl_train import make_groups, parse_args
    old = sys.argv
    sys.argv = ["x", "--run-name", "t", "--configs", "line3:T2", "--n-envs", "4", "--device", "cpu"]
    try:
        args = parse_args()
    finally:
        sys.argv = old
    assert args.residual is False
    groups, _ = make_groups(args, "cpu")
    assert type(groups[0][1].env) is ModularEnv and groups[0][1].obs_dim == OBS_DIM
    sys.argv = ["x", "--run-name", "t", "--configs", "line3:T2", "--n-envs", "4", "--device", "cpu", "--residual"]
    try:
        args = parse_args()
    finally:
        sys.argv = old
    groups, _ = make_groups(args, "cpu")
    assert type(groups[0][1].env) is ResidualEnv
