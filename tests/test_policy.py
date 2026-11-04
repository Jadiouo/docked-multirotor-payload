"""ModularActorCritic 的單元測試（CPU 即可）：置換等變、k 跳局部性、active mask、act 與 evaluate 一致。"""
import pytest
import torch

from quadsim.assembly import CONFIGS
from quadsim.env import ModularEnv
from quadsim.policy import ModularActorCritic, PackLayout, OBS_DIM
from quadsim.rl_env import PackedEnv

torch.set_num_threads(1)


def line_adj(n):
    """n 個模組排成一條線（+x 方向），adj[i, j, k] = j 是 i 在方向 k 的鄰居；k = +x, −x, +y, −y。"""
    adj = torch.zeros(n, n, 4)
    for i in range(n - 1):
        adj[i, i + 1, 0] = 1       # i+1 在 i 的 +x
        adj[i + 1, i, 1] = 1       # i 在 i+1 的 −x
    return adj


def make_policy(n_max, k, hist=1, **kw):
    torch.manual_seed(0)
    # head_std=1：預設的 0.01 會讓輸出幾乎是常數，測試就沒有鑑別力
    return ModularActorCritic(n_max, hist=hist, k=k, head_std=1.0, **kw).eval()


def hops(adj):
    """從模組 0 出發的最短躍點數（BFS），不連通 = inf。"""
    n = adj.shape[0]
    reach = (adj.sum(-1) > 0) | (adj.sum(-1).T > 0)
    dist = [float("inf")] * n; dist[0] = 0; frontier = [0]
    while frontier:
        nxt = []
        for i in frontier:
            for j in range(n):
                if reach[i, j] and dist[j] == float("inf"):
                    dist[j] = dist[i] + 1; nxt.append(j)
        frontier = nxt
    return dist


@pytest.mark.parametrize("k", [0, 1, 2])
def test_permutation_equivariance(k):
    n, B = 5, 3
    pol = make_policy(n, k)
    adj = line_adj(n); adj[0, 3, 2] = adj[3, 0, 3] = 1          # 加一條 y 方向的邊，讓拓樸不只是一條線
    x = torch.randn(B, n, OBS_DIM); active = torch.ones(B, n)
    perm = torch.randperm(n)
    out = pol.actor_forward(x, adj, active)
    out_p = pol.actor_forward(x[:, perm], adj[perm][:, perm], active[:, perm])     # 重排模組，鄰接的行列一起重排
    torch.testing.assert_close(out_p, out[:, perm], atol=1e-5, rtol=1e-5)
    # critic 同理（每個模組的 value 跟著重排）
    xc = torch.randn(B, n, OBS_DIM + 4)
    torch.testing.assert_close(pol.critic_forward(xc[:, perm], adj[perm][:, perm], active[:, perm]),
                               pol.critic_forward(xc, adj, active)[:, perm], atol=1e-5, rtol=1e-5)
    if k > 0:    # 方向的語意有被用到：鄰居在 +x 或 −x 輸出不同（避免「對方向不敏感」也能通過等變測試）
        assert not torch.allclose(out, pol.actor_forward(x, adj.flip(-1), active), atol=1e-4)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_k_hop_locality(k):
    n = 8
    pol = make_policy(n, k)
    adj, active = line_adj(n), torch.ones(1, n)
    x = torch.randn(1, n, OBS_DIM)
    base = pol.actor_forward(x, adj, active)[0, 0]
    d = hops(adj)
    far = [j for j in range(n) if d[j] > k]            # 比 k 跳更遠的模組
    near = [j for j in range(n) if d[j] <= k and j != 0]
    x2 = x.clone(); x2[:, far] = torch.randn(1, len(far), OBS_DIM) * 5
    torch.testing.assert_close(pol.actor_forward(x2, adj, active)[0, 0], base, atol=1e-6, rtol=0)
    if near:    # 對照：k 跳以內的模組改了，輸出要變（否則上面的測試沒有意義）
        x3 = x.clone(); x3[:, near[-1]] += 1.0
        assert not torch.allclose(pol.actor_forward(x3, adj, active)[0, 0], base, atol=1e-4)
    # 剛好 k 跳以外的第一個模組（k+1 跳）
    if k + 1 < n:
        x4 = x.clone(); x4[:, k + 1] += 3.0
        torch.testing.assert_close(pol.actor_forward(x4, adj, active)[0, 0], base, atol=1e-6, rtol=0)


def test_active_mask_isolates_padding():
    n = 5
    pol = make_policy(n, 2)
    adj = line_adj(n)
    adj[3, 4, 0] = adj[4, 3, 1] = 0; adj[2, 3, 0] = adj[3, 2, 1] = 1     # 前 4 個真模組；第 5 個是假的
    active = torch.tensor([[1., 1, 1, 1, 0]])
    x = torch.randn(1, n, OBS_DIM)
    out = pol.actor_forward(x, adj, active)
    x2 = x.clone(); x2[:, 4] = torch.randn(OBS_DIM) * 50                 # 假模組的 obs 亂改
    out2 = pol.actor_forward(x2, adj, active)
    torch.testing.assert_close(out2, out, atol=1e-6, rtol=0)
    assert (out[:, 4] == 0).all()                                        # 假模組自己的輸出為 0
    # 和「真的只有 4 個模組」的 policy 輸出一致（同一組權重）
    pol4 = make_policy(4, 2); pol4.load_state_dict({k_: v for k_, v in pol.state_dict().items()}, strict=True)
    adj4 = adj[:4, :4].clone(); adj4[3, :, :] = adj[3, :4]
    torch.testing.assert_close(pol4.actor_forward(x[:, :4], adj4, active[:, :4]), out[:, :4], atol=1e-5, rtol=1e-5)
    # critic 的 mean pool 也不能把假模組算進去
    xc = torch.randn(1, n, OBS_DIM + 4)
    xc2 = xc.clone(); xc2[:, 4] = 100.0
    torch.testing.assert_close(pol.critic_forward(xc2, adj, active), pol.critic_forward(xc, adj, active), atol=1e-6, rtol=0)


def test_act_evaluate_consistent_and_ppo_shapes():
    n, hist, B = 4, 3, 6
    pol = make_policy(n, 1, hist=hist)
    torch.manual_seed(1)
    env = ModularEnv(CONFIGS["line3"], B, "T1", seed=0)
    pe = PackedEnv(env, n, hist)
    obs, cobs = pe.reset()
    assert obs.shape == (B, n, pol.actor_layout.size) and cobs.shape == (B, n, pol.critic_layout.size)
    for _ in range(3):                       # 走幾步，讓歷史裡有不同的 obs
        action, logprob = pol.act(obs)
        obs, cobs, *_ = pe.step(action)
    action, logprob = pol.act(obs)
    assert action.shape == (B, n, 4) and logprob.shape == (B, n)
    lp, ent = pol.evaluate(obs, action)
    torch.testing.assert_close(lp, logprob, atol=1e-5, rtol=1e-5)
    assert ent.shape == (B, n) and torch.isfinite(ent).all()
    assert pol.value(cobs).shape == (B, n)
    det, _ = pol.act(obs, deterministic=True)
    torch.testing.assert_close(det, pol.act(obs, deterministic=True)[0])
    # 假模組（第 4 個）：logprob 有限，輸出動作 mean 為 0（不會產生 NaN 汙染 ppo 的 masked 平均）
    assert torch.isfinite(logprob).all()


def test_history_wrapper_and_final_obs():
    n, hist, B = 2, 3, 4
    env = ModularEnv(CONFIGS["line2"], B, "T1", seed=0)
    pe = PackedEnv(env, n, hist)
    pe.reset()
    obs0 = pe.hist_buf.clone()
    assert torch.equal(obs0[:, :, 0], obs0[:, :, -1])                    # 開頭：歷史用第一個 obs 填滿
    u = torch.zeros(B, n, 4)
    _, _, _, term, trunc, info = pe.step(u)
    # final_obs 的最新一格 = 沒結束的環境此刻的 obs（也就是新歷史的最後一格）
    keep = ~(term | trunc)
    final_x = pe.actor_layout.unpack(info["final_obs"])[0][:, 0]         # [B, n_max(模組), H*28]
    new_x = pe.actor_layout.unpack(pe._pack_actor(pe.hist_buf))[0][:, 0]
    torch.testing.assert_close(final_x[keep], new_x[keep])
    # 沒結束的環境：歷史往前滑一格
    torch.testing.assert_close(pe.hist_buf[keep][:, :, :-1], obs0[keep][:, :, 1:])
    # 結束的環境：新歷史用新 episode 的第一個 obs 填滿
    if (~keep).any():
        h = pe.hist_buf[~keep]
        assert torch.equal(h[:, :, 0], h[:, :, -1])


def test_history_reset_on_done():
    n, hist, B = 2, 3, 4
    env = ModularEnv(CONFIGS["line2"], B, "T1", seed=0)
    pe = PackedEnv(env, n, hist)
    pe.reset()
    env.set_state(idx=[1], pos=[[0., 0., -1.0]])          # 讓第 1 個環境墜地 → terminated
    _, _, _, term, trunc, info = pe.step(torch.zeros(B, n, 4))
    assert term[1] and not term[0]
    h = pe.hist_buf[1]
    assert torch.equal(h[:, 0], h[:, -1])                  # 新 episode：歷史用新的第一個 obs 填滿
    x = pe.actor_layout.unpack(info["final_obs"])[0][1, 0].reshape(n, hist, OBS_DIM)
    assert not torch.equal(x[:, -1], x[:, 0])              # final_obs 仍是 reset 之前的歷史（最新一格 ≠ 最舊一格）
    # final_obs 最新一格的 acc/姿態等來自墜地那一步，而不是新 episode（e_p 的 z 分量差很多）
    assert not torch.allclose(x[:, -1, :3], h[:, -1, :3])


def test_team_entry_points_match_per_row():
    """act_team / value_team（整隊只算一次 GNN）要與逐列的 act / value 數值一致，包含有補齊的假模組。"""
    n, hist, B = 4, 2, 5
    pol = make_policy(n, 1, hist=hist)
    torch.manual_seed(3)
    env = ModularEnv(CONFIGS["line3"], B, "T1", seed=0)          # 3 個真模組 + 1 個假模組
    pe = PackedEnv(env, n, hist)
    obs, cobs = pe.reset()
    for _ in range(3):
        action, _ = pol.act(obs)
        obs, cobs, *_ = pe.step(action)
    a_row, lp_row = pol.act(obs, deterministic=True)
    a_team, lp_team0 = pol.act_team(obs, deterministic=True)
    torch.testing.assert_close(a_team, a_row, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(lp_team0, lp_row, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(pol.value_team(cobs), pol.value(cobs), atol=1e-6, rtol=1e-5)
    # 隨機取樣：用 team 版取樣出的動作，逐列 evaluate 的 logprob 要等於 team 版回報的 logprob
    a_s, lp_s = pol.act_team(obs)
    lp_eval, _ = pol.evaluate(obs, a_s)
    torch.testing.assert_close(lp_eval, lp_s, atol=1e-5, rtol=1e-5)
