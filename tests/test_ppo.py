import numpy as np
import pytest
import torch

from ppo import (ActorCritic, PPOConfig, RolloutBuffer, RunningMeanStd,
                 compute_gae, ppo_loss, ppo_update)

torch.set_num_threads(2)


# ---------------------------------------------------------------- GAE
def gae_reference(r, v, nv, term, trunc, gamma, lam):
    """獨立寫的純 Python 參考實作：一條時間序列，一次只看一步。
    用「往後累加」的定義：A_t = Σ_k (γλ)^k δ_{t+k}，遇到 episode 結束就停止累加。"""
    T = len(r)
    adv = []
    for t in range(T):
        total, discount = 0.0, 1.0
        for k in range(t, T):
            boot = 0.0 if term[k] else nv[k]
            total += discount * (r[k] + gamma * boot - v[k])
            if term[k] or trunc[k]:
                break
            discount *= gamma * lam
        adv.append(total)
    return np.array(adv), np.array(adv) + np.array(v)


def random_sequence(T, rng, term_p, trunc_p):
    r, v, nv = rng.normal(size=(3, T))
    term = rng.random(T) < term_p
    trunc = (rng.random(T) < trunc_p) & ~term  # 同一步不會同時 terminated 又 truncated
    return r, v, nv, term, trunc


@pytest.mark.parametrize("term_p,trunc_p", [(0.2, 0.0), (0.0, 0.2), (0.15, 0.15), (0.0, 0.0)])
def test_gae_matches_reference(term_p, trunc_p):
    rng = np.random.default_rng(0)
    r, v, nv, term, trunc = random_sequence(30, rng, term_p, trunc_p)
    adv, ret = compute_gae(*[torch.tensor(x, dtype=torch.float64) for x in (r, v, nv)],
                           torch.tensor(term), torch.tensor(trunc), 0.97, 0.9)
    adv_ref, ret_ref = gae_reference(r, v, nv, term, trunc, 0.97, 0.9)
    np.testing.assert_allclose(adv.numpy(), adv_ref, atol=1e-10)
    np.testing.assert_allclose(ret.numpy(), ret_ref, atol=1e-10)


def test_gae_terminated_vs_truncated_by_hand():
    # 單步 episode：terminated 不 bootstrap，truncated 要 bootstrap
    z = torch.zeros(1)
    one = torch.ones(1)
    adv_term, _ = compute_gae(one, z, 5 * one, one, z, 0.9, 0.95)
    adv_trunc, _ = compute_gae(one, z, 5 * one, z, one, 0.9, 0.95)
    assert adv_term.item() == pytest.approx(1.0)
    assert adv_trunc.item() == pytest.approx(1.0 + 0.9 * 5.0)


def test_gae_shapes_and_flatten_consistency():
    # [T, n_envs, n_agents] 的結果 == 攤平成 [T, n_envs*n_agents] == 逐 agent 分開算
    T, E, A = 12, 3, 4
    rng = np.random.default_rng(1)
    r, v, nv = (torch.tensor(rng.normal(size=(T, E, A)), dtype=torch.float64) for _ in range(3))
    term = torch.tensor(rng.random((T, E, A)) < 0.1)
    trunc = torch.tensor(rng.random((T, E, A)) < 0.1) & ~term
    adv, ret = compute_gae(r, v, nv, term, trunc, 0.99, 0.95)
    assert adv.shape == ret.shape == (T, E, A)

    flat = lambda x: x.reshape(T, E * A)
    adv_f, _ = compute_gae(*map(flat, (r, v, nv, term, trunc)), 0.99, 0.95)
    torch.testing.assert_close(adv.reshape(T, E * A), adv_f)

    for a in range(A):
        adv_a, _ = compute_gae(r[..., a], v[..., a], nv[..., a], term[..., a], trunc[..., a], 0.99, 0.95)
        torch.testing.assert_close(adv[..., a], adv_a)


def test_gae_shape_mismatch_asserts():
    T, E, A = 5, 2, 3
    x = torch.zeros(T, E, A)
    done_per_env = torch.zeros(T, E, dtype=torch.bool)  # 忘了 expand 到 agent 維度
    with pytest.raises(AssertionError):
        compute_gae(x, x, x, done_per_env, x.bool(), 0.99, 0.95)
    with pytest.raises(AssertionError):
        compute_gae(x, x, x[:-1], x.bool(), x.bool(), 0.99, 0.95)


# ---------------------------------------------------------------- RunningMeanStd
def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(2)
    rms = RunningMeanStd(shape=(3,))
    chunks = [rng.normal(loc=2.0, scale=3.0, size=(n, 3)) for n in (5, 17, 1, 40)]
    for c in chunks:
        rms.update(torch.tensor(c))
    allx = np.concatenate(chunks)
    np.testing.assert_allclose(rms.mean.numpy(), allx.mean(0), atol=1e-10)
    np.testing.assert_allclose(rms.var.numpy(), allx.var(0), atol=1e-10)
    assert rms.count == len(allx)


def test_running_mean_std_leading_dims_and_inverse():
    rng = np.random.default_rng(3)
    x = torch.tensor(rng.normal(loc=-1.0, scale=2.0, size=(6, 4, 2)), dtype=torch.float32)  # [T,E,dim]
    rms = RunningMeanStd(shape=(2,))
    rms.update(x)
    np.testing.assert_allclose(rms.mean.numpy(), x.reshape(-1, 2).double().mean(0).numpy(), atol=1e-6)
    torch.testing.assert_close(rms.denormalize(rms.normalize(x)), x, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rms.normalize(rms.denormalize(x)), x, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------- 小 buffer
def make_buffer(T=8, E=3, A=2, obs_dim=4, act_dim=2, seed=0, critic_obs_dim=None):
    torch.manual_seed(seed)
    agent = ActorCritic(obs_dim, act_dim, critic_obs_dim)
    buf = RolloutBuffer(T, (E, A), obs_dim, act_dim, critic_obs_dim)
    for t in range(T):
        obs = torch.randn(E, A, obs_dim)
        cobs = torch.randn(E, A, buf.critic_obs.shape[-1])
        action, logprob = agent.act(obs)
        term = torch.rand(E, A) < 0.1
        trunc = (torch.rand(E, A) < 0.1) & ~term
        buf.add(t, obs, cobs, action, logprob, torch.randn(E, A), agent.value(cobs),
                term, trunc, agent.value(torch.randn(E, A, buf.critic_obs.shape[-1])))
    buf.compute_advantages(0.99, 0.95)
    return agent, buf


def flat_batch(buf):
    N = buf.rewards.numel()
    return {k: getattr(buf, k).reshape(N, *getattr(buf, k).shape[buf.rewards.ndim:])
            for k in ["obs", "critic_obs", "actions", "logprobs", "values",
                      "advantages", "returns", "active"]}


def test_active_mask_ignores_masked_elements():
    cfg = PPOConfig()
    agent, buf = make_buffer(critic_obs_dim=5)
    buf.active[:, :, 1] = 0.0          # 第 1 個 agent 整個被遮掉（像補齊的空位）
    buf.active[2, 0, 0] = 0.0          # 再單獨遮掉一個元素

    def loss_of(b):
        return {k: float(v.detach()) for k, v in ppo_loss(agent, flat_batch(b), cfg).items()}

    base = loss_of(buf)
    # (a) 改被遮掉 agent 的 reward，重算 GAE：loss 不變
    buf.rewards[:, :, 1] += 100.0
    buf.compute_advantages(0.99, 0.95)
    assert loss_of(buf) == pytest.approx(base, rel=1e-6)
    # (b) 直接改被遮掉元素的 advantage / returns / obs：loss 不變
    buf.advantages[:, :, 1] = 1e3
    buf.advantages[2, 0, 0] = -1e3
    buf.returns[2, 0, 0] = 1e3
    buf.obs[:, :, 1] = 7.0
    assert loss_of(buf) == pytest.approx(base, rel=1e-6)
    # 對照組：改 active=1 的元素，loss 一定會變
    buf.advantages[0, 0, 0] += 5.0
    assert loss_of(buf)["policy_loss"] != pytest.approx(base["policy_loss"], rel=1e-6)


def test_ppo_update_returns_stats_and_changes_params():
    cfg = PPOConfig(n_epochs=2, n_minibatches=4)
    agent, buf = make_buffer(critic_obs_dim=5)
    opt = torch.optim.Adam(agent.parameters(), lr=1e-2)
    before = [p.detach().clone() for p in agent.parameters()]
    stats = ppo_update(agent, opt, buf, cfg)
    for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
              "explained_variance", "grad_norm"]:
        assert k in stats and np.isfinite(stats[k]), k
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.parameters()))
    assert buf.advantages.shape == buf.rewards.shape  # 未正規化的 advantages 還在


def test_ppo_update_with_value_rms_and_unclipped_vloss():
    cfg = PPOConfig(n_epochs=1, n_minibatches=2, clip_vloss=False, norm_adv=False)
    agent, buf = make_buffer()
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    stats = ppo_update(agent, opt, buf, cfg, value_rms=RunningMeanStd())
    assert np.isfinite(stats["value_loss"])


# ---------------------------------------------------------------- 追加的測試
def test_gae_terminated_and_truncated_both_set():
    # 同一步兩者都是 1：不 bootstrap（terminated 優先），鏈也斷開
    T = 3
    r, v, nv = torch.ones(T), torch.full((T,), 0.5), torch.full((T,), 7.0)
    term = torch.tensor([0.0, 1.0, 0.0])
    trunc = torch.tensor([0.0, 1.0, 0.0])
    adv, _ = compute_gae(r, v, nv, term, trunc, 0.9, 0.95)
    assert adv[1].item() == pytest.approx(1.0 - 0.5)           # δ_1 = r - v，沒有 bootstrap
    ref, _ = gae_reference(r.numpy(), v.numpy(), nv.numpy(), term.bool().numpy(), trunc.bool().numpy(), 0.9, 0.95)
    np.testing.assert_allclose(adv.numpy(), ref, atol=1e-6)


class StateStdAC(ActorCritic):
    """log_std 隨 obs 變化，讓每個樣本的 entropy 都不同（否則測不出 mask 有沒有作用）。"""

    def __init__(self, obs_dim, act_dim, critic_obs_dim=None):
        super().__init__(obs_dim, act_dim, critic_obs_dim)
        self.std_head = torch.nn.Linear(obs_dim, act_dim)

    def _dist(self, obs):
        mean = self.actor_mean(obs)
        return torch.distributions.Normal(mean, self.std_head(obs).clamp(-2, 1).exp())


def test_mask_subset_equivalence():
    torch.manual_seed(0)
    agent = StateStdAC(4, 2, 5)
    _, buf = make_buffer(critic_obs_dim=5)
    batch = flat_batch(buf)
    batch["logprobs"] = batch["logprobs"] + 0.1 * torch.randn_like(batch["logprobs"])  # ratio != 1
    mask = (torch.rand(batch["active"].shape) < 0.5).float()
    mask[:2] = 1.0
    batch["active"] = mask
    sub = {k: v[mask > 0] for k, v in batch.items()}
    sub["active"] = torch.ones_like(sub["active"])
    cfg = PPOConfig()
    full, part = ppo_loss(agent, batch, cfg), ppo_loss(agent, sub, cfg)
    for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]:
        assert full[k].item() == pytest.approx(part[k].item(), rel=1e-5), k
    # 熵在不同樣本間確實不同，這個比對才有意義
    _, ent = agent.evaluate(batch["obs"], batch["actions"])
    assert ent.std() > 1e-3


def test_all_zero_mask_is_finite():
    agent, buf = make_buffer()
    batch = flat_batch(buf)
    batch["active"] = torch.zeros_like(batch["active"])
    batch["advantages"] = torch.full_like(batch["advantages"], float("nan"))  # padding 可能是 NaN
    batch["returns"] = torch.full_like(batch["returns"], float("inf"))
    out = ppo_loss(agent, batch, PPOConfig())
    assert all(torch.isfinite(v) for v in out.values())


class StubAgent:
    """evaluate / value 直接回傳預先指定的數值，用來手算驗證 ppo_loss。"""

    def __init__(self, logprob, entropy, value):
        self.l, self.e, self.v = logprob, entropy, value

    def evaluate(self, obs, action):
        return self.l, self.e

    def value(self, critic_obs):
        return self.v


def test_ppo_loss_by_hand():
    ratio = torch.tensor([1.5, 0.5, 1.1, 0.9])
    old_lp = torch.zeros(4)
    batch = {"obs": torch.zeros(4, 1), "critic_obs": torch.zeros(4, 1), "actions": torch.zeros(4, 1),
             "logprobs": old_lp, "values": torch.zeros(4), "advantages": torch.tensor([1., 1., -1., -1.]),
             "returns": torch.tensor([0., 0.3, 0.1, 1.0]), "active": torch.ones(4)}
    agent = StubAgent(ratio.log(), torch.tensor([1., 2., 3., 4.]), torch.tensor([0.5, -0.5, 0.1, 0.8]))
    cfg = PPOConfig(norm_adv=False, clip_vloss=True, clip_coef=0.2, vf_coef=0.5, ent_coef=0.01)
    out = ppo_loss(agent, batch, cfg)
    # policy：max(-A·r, -A·clip(r)) = [-1.2, -0.5, 1.1, 0.9]，平均 0.075
    assert out["policy_loss"].item() == pytest.approx(0.075)
    # value：未 clip 的平方誤差 [.25,.64,0,.04]，clip 後 V=[.2,-.2,.1,.2]（第 4 個 .8 被夾到 .2）→ [.04,.25,0,.64]
    #       取 max = [.25,.64,0,.64]，0.5·平均 = 0.19125
    assert out["value_loss"].item() == pytest.approx(0.19125)
    assert out["entropy"].item() == pytest.approx(2.5)
    assert out["clip_frac"].item() == pytest.approx(0.5)
    r = ratio.numpy()
    assert out["approx_kl"].item() == pytest.approx(np.mean((r - 1) - np.log(r)), rel=1e-5)
    assert out["loss"].item() == pytest.approx(0.075 + 0.5 * 0.19125 - 0.01 * 2.5)
    # 關掉 value clip：直接是未 clip 版本 0.5·mean([.25,.64,0,.04]) = 0.11625
    out2 = ppo_loss(agent, batch, PPOConfig(norm_adv=False, clip_vloss=False))
    assert out2["value_loss"].item() == pytest.approx(0.11625)


def test_first_minibatch_ratio_is_one():
    # 一次更新前 π_new == π_old：ratio = 1 → clip_frac = 0、approx_kl ≈ 0
    agent, buf = make_buffer()
    opt = torch.optim.SGD(agent.parameters(), lr=0.0)
    stats = ppo_update(agent, opt, buf, PPOConfig(n_epochs=1, n_minibatches=1))
    assert stats["clip_frac"] == 0.0
    assert abs(stats["approx_kl"]) < 1e-6


def test_adv_norm_scope_buffer():
    agent, buf = make_buffer()
    buf.active[:, :, 1] = 0.0
    before = buf.advantages.clone()
    opt = torch.optim.SGD(agent.parameters(), lr=0.0)
    stats = ppo_update(agent, opt, buf, PPOConfig(n_epochs=1, n_minibatches=3, adv_norm_scope="buffer"))
    assert np.isfinite(stats["policy_loss"])
    torch.testing.assert_close(buf.advantages, before)  # buffer 裡保留未正規化版本
    with pytest.raises(ValueError):
        ppo_update(agent, opt, buf, PPOConfig(adv_norm_scope="oops"))


def test_ppo_update_flatten_alignment():
    T, E, A = 5, 3, 4
    N = T * E * A

    class Recorder(ActorCritic):
        def __init__(self):
            super().__init__(2, 2, 2)
            self.seen = []   # 每次 (obs, actions, critic_obs) 的第 0 欄（編碼的索引）

        def evaluate(self, obs, action):
            self._last = (obs[:, 0].detach().clone(), action[:, 0].detach().clone())
            return super().evaluate(obs, action)

        def value(self, critic_obs):
            self.seen.append((*self._last, critic_obs[:, 0].detach().clone()))
            return super().value(critic_obs)

    agent = Recorder()
    buf = RolloutBuffer(T, (E, A), 2, 2, 2)
    idx = torch.arange(N, dtype=torch.float32).reshape(T, E, A)   # 每個 [t,e,a] 編碼自己的索引
    for t in range(T):
        code = idx[t].unsqueeze(-1).expand(E, A, 2)
        buf.add(t, code, code, code, torch.zeros(E, A), torch.zeros(E, A), torch.zeros(E, A),
                torch.zeros(E, A), torch.zeros(E, A), torch.zeros(E, A))
    buf.compute_advantages(0.99, 0.95)
    ppo_update(agent, torch.optim.SGD(agent.parameters(), lr=0.0), buf,
               PPOConfig(n_epochs=1, n_minibatches=4))
    for o, a, c in agent.seen:               # 同一個樣本的 obs / actions / critic_obs 索引一致
        assert torch.equal(o, a) and torch.equal(o, c)
    all_idx = torch.cat([o for o, _, _ in agent.seen])
    assert torch.equal(all_idx.sort().values, torch.arange(N, dtype=torch.float32))  # 每個樣本恰好一次
