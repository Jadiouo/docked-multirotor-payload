"""mean_bound_coef（動作均值越界懲罰）：coef=0 與舊版 loss 一致；coef>0 時越界的 μ 得到把它拉回界內的梯度。"""
import torch

from ppo import PPOConfig, ppo_loss
from quadsim.assembly import CONFIGS
from quadsim.env import ModularEnv
from quadsim.policy import ModularActorCritic
from quadsim.rl_env import PackedEnv

torch.set_num_threads(1)


def make(head_std=3.0):
    torch.manual_seed(0)
    env = ModularEnv(CONFIGS["single"], 8, task="T2", seed=0)
    pe = PackedEnv(env, 2, 1)
    agent = ModularActorCritic(2, hist=1, k=1, head_std=head_std)
    obs, cobs = pe.reset()
    n = 8 * 2
    o, c = obs.reshape(n, -1), cobs.reshape(n, -1)
    with torch.no_grad():
        a, lp = agent.act(o)
        v = agent.value(c)
    adv = torch.randn(n)
    batch = dict(obs=o, critic_obs=c, actions=a, logprobs=lp, values=v, advantages=adv, returns=v + adv,
                 active=torch.ones(n))
    return agent, batch


def test_coef_zero_matches_default():
    agent, batch = make()
    base = ppo_loss(agent, batch, PPOConfig())
    zero = ppo_loss(agent, batch, PPOConfig(mean_bound_coef=0.0))
    assert torch.equal(base["loss"], zero["loss"]) and "mean_pen" not in zero


def test_penalty_value_and_gradient_pulls_back():
    agent, batch = make()
    cfg = PPOConfig(mean_bound_coef=1.0)
    out = ppo_loss(agent, batch, cfg)
    mu = agent.last_mean
    assert (mu.abs() > 1).any(), "測試要有越界的 μ（head_std 放大）"
    want = (torch.relu(mu.abs() - 1) ** 2).mean(-1).mean()
    assert torch.allclose(out["mean_pen"], want.detach()) and out["mean_oob"] > 0
    base = ppo_loss(agent, batch, PPOConfig())
    assert torch.allclose(out["loss"], base["loss"] + out["mean_pen"], atol=1e-6)
    # 純懲罰項的梯度：沿負梯度走一步，越界量變小
    agent.zero_grad()
    ppo_loss(agent, batch, cfg)["loss"].backward()
    g_total = torch.cat([p.grad.flatten() for p in agent.parameters() if p.grad is not None])
    # 只看懲罰：對 μ 的梯度方向 = sign(μ)·(|μ|-1)+，用 autograd 直接驗證
    mu = agent.last_mean.detach().requires_grad_(True)
    pen = (torch.relu(mu.abs() - 1) ** 2).mean(-1).mean(); pen.backward()
    oob = mu.abs() > 1
    assert (mu.grad[oob] * mu.detach()[oob] > 0).all()           # 梯度與 μ 同號 → 梯度下降會把 |μ| 往內拉
    assert (mu.grad[~oob] == 0).all() and g_total.abs().sum() > 0
    # 實際走一步參數更新後，越界量下降
    agent2, batch2 = make()
    opt = torch.optim.SGD(agent2.parameters(), lr=1e-4)
    cfg2 = PPOConfig(mean_bound_coef=10.0)
    pen0 = ppo_loss(agent2, batch2, cfg2)["mean_pen"].item()
    for _ in range(5):
        opt.zero_grad(); ppo_loss(agent2, batch2, cfg2)["loss"].backward(); opt.step()
    assert ppo_loss(agent2, batch2, cfg2)["mean_pen"].item() < pen0
