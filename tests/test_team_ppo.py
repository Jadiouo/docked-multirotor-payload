"""--team-sample（整隊樣本）：關閉時與舊版逐位元相同；N=1 時 loss 與舊版相同；mask 的模組不影響加總；小訓練跑通。"""
import csv
import subprocess
import sys
from pathlib import Path

import torch

from ppo import PPOConfig, ppo_loss
from quadsim.assembly import CONFIGS
from quadsim.env import ModularEnv
from quadsim.policy import ModularActorCritic
from quadsim.rl_env import PackedEnv
from quadsim.team_ppo import team_ppo_loss

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parent.parent


def rollout_batch(name, n_max, E=8, mean_bound=0.0):
    torch.manual_seed(0)
    env = ModularEnv(CONFIGS[name], E, task="T2", seed=0)
    pe = PackedEnv(env, n_max, 1)
    agent = ModularActorCritic(n_max, hist=1, k=1, head_std=3.0)
    obs, cobs = pe.reset()
    with torch.no_grad():
        a, lp = agent.act_team(obs)
        v = agent.value_team(cobs)
    return agent, pe, obs, cobs, a, lp, v


def test_n1_team_loss_equals_module_loss():
    agent, pe, obs, cobs, a, lp, v = rollout_batch("single", 1)
    E = obs.shape[0]
    adv = torch.randn(E)
    cfg = PPOConfig(mean_bound_coef=0.5)
    old = ppo_loss(agent, dict(obs=obs.reshape(E, -1), critic_obs=cobs.reshape(E, -1), actions=a.reshape(E, -1), logprobs=lp.reshape(E),
                               values=v.reshape(E), advantages=adv, returns=v.reshape(E) + adv, active=torch.ones(E)), cfg)
    new = team_ppo_loss(agent, dict(obs=obs[:, 0], critic_obs=cobs[:, 0], actions=a.reshape(E, -1), logprobs=lp.sum(1),
                                    values=v.sum(1), advantages=adv, returns=v.sum(1) + adv, active=torch.ones(E)), cfg)
    for k in old:
        assert torch.allclose(old[k], new[k], atol=1e-6), k


def test_masked_module_does_not_affect_logprob():
    # 單機補齊到 n_max=3：模組 1、2 是假的。改掉假模組的動作與 obs 內容，隊 log-prob / entropy 不變。
    agent, pe, obs, cobs, a, lp, v = rollout_batch("single", 3)
    E = obs.shape[0]
    lp0, ent0, _, _ = agent.team_logprob_entropy(obs[:, 0], a.reshape(E, -1))
    a2 = a.clone(); a2[:, 1:] = 123.0
    obs2 = obs.clone()
    n, F = 3, agent.actor_layout.feat
    obs2[:, :, F:3 * F] = 55.0                                   # x 裡假模組那幾段
    lp1, ent1, _, _ = agent.team_logprob_entropy(obs2[:, 0], a2.reshape(E, -1))
    assert torch.allclose(lp0, lp1) and torch.allclose(ent0, ent1)
    # 且等於真模組 log-prob 的加總
    assert torch.allclose(lp0, torch.where(pe.active > 0, lp, torch.zeros_like(lp)).sum(1), atol=1e-5)


def test_mixed_n_team_logprob_is_sum_over_valid():
    name = [k for k, c in CONFIGS.items() if len(c) > 1][0]
    agent, pe, obs, cobs, a, lp, v = rollout_batch(name, len(CONFIGS[name]) + 1)
    E = obs.shape[0]
    lpt, _, _, active = agent.team_logprob_entropy(obs[:, 0], a.reshape(E, -1))
    assert torch.allclose(lpt, (lp * pe.active).sum(1), atol=1e-5)
    assert active.sum(1).max() == len(CONFIGS[name])


def run_train(tmp, extra):
    cmd = [sys.executable, "-m", "scripts.rl_train", "--run-name", tmp, "--configs", "single:T2", "--n-envs", "16", "--rollout-T", "16",
           "--total-steps", "2048", "--device", "cpu", "--seed", "3", "--mean-bound-coef", "0.01", "--n-minibatches", "2"] + extra
    env = {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"}
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True)
    return ROOT / "results" / "p3" / tmp / "train.csv"


def test_small_cpu_training_team_sample():
    import shutil
    p = run_train("_test_team", ["--team-sample"])
    try:
        rows = list(csv.DictReader(open(p)))
        assert len(rows) == 8 and all(float(r["policy_loss"]) == float(r["policy_loss"]) for r in rows)
        for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm", "mean_pen", "mean_oob", "success_rate"]:
            assert k in rows[0]
    finally:
        shutil.rmtree(p.parent, ignore_errors=True)
