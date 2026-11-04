"""手寫 PPO 核心（單一檔案，只依賴 torch 和 numpy）。介面規格見 docs/ppo_interface.md（v0 + v0.1）。

張量的前導維度不寫死：rollout 的形狀是 [T, *B]，B 可以是 (n_envs,) 或 (n_envs, n_agents)。
GAE 沿第 0 維（時間）做，minibatch 更新時把 [T, *B] 攤平成一條 N = T * prod(B) 的資料。

內容（由上到下）：
  compute_gae      純函式，GAE(λ)，分開處理 terminated / truncated
  RunningMeanStd   running mean/std，用來正規化 obs、return、value
  ActorCritic      預設的 Gaussian MLP（仿 CleanRL ppo_continuous_action）
  RolloutBuffer    存 rollout 的欄位，含可選的 active mask
  PPOConfig        超參數
  ppo_loss         單一 minibatch 的 PPO loss
  ppo_update       n_epochs × n_minibatches 次更新，回傳統計量
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# 1) GAE
# ============================================================================
def compute_gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    """廣義優勢估計 GAE(λ)。所有輸入形狀必須完全相同：[T, *B]（不自動廣播）。

    next_values[t] = V(s_{t+1})，由呼叫端算好。若 s_{t+1} 是 episode 結束後 reset 的新 obs，
    呼叫端要改用 reset 之前的 final obs 來算（autoreset 回傳的 obs 不能拿來 bootstrap）。

    對每個 t（從後往前）：
        δ_t = r_t + γ · (1 - terminated_t) · V(s_{t+1}) - V(s_t)
        A_t = δ_t + γ · λ · (1 - done_t) · A_{t+1},     done_t = terminated_t OR truncated_t
        R_t = A_t + V(s_t)                              （λ-return，當作 value 的目標）

    - terminated：真正的終止狀態，未來回報為 0，所以 next_value 乘 0。
    - truncated：撞到 time limit，狀態本身沒有終止，所以仍用 V(s_{t+1}) bootstrap；
      但 t+1 屬於下一個 episode，所以 A_{t+1} 不能傳回來，鏈在這裡斷開（done_t = 1）。
    回傳 advantages, returns，形狀都是 [T, *B]。
    """
    shape = rewards.shape
    for name, x in [("values", values), ("next_values", next_values),
                    ("terminated", terminated), ("truncated", truncated)]:
        assert x.shape == shape, (
            f"compute_gae 不會自動廣播：{name} 的形狀 {tuple(x.shape)} 和 rewards 的 "
            f"{tuple(shape)} 不一致，請呼叫端先 expand")

    terminated = terminated.to(rewards.dtype)
    truncated = truncated.to(rewards.dtype)
    done = torch.clamp(terminated + truncated, max=1.0)  # 兩者任一發生，episode 就在這步結束

    T = shape[0]
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros_like(rewards[0])  # A_{t+1}，最後一步之後沒有東西，設為 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * (1.0 - terminated[t]) * next_values[t] - values[t]
        last_adv = delta + gamma * lam * (1.0 - done[t]) * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


# ============================================================================
# 2) RunningMeanStd
# ============================================================================
class RunningMeanStd:
    """用平行合併公式（Chan et al.）累積 mean / var。可用來正規化 obs、return、value。

    update(x) 的 x 形狀為 [..., *shape]，前面所有維度都當成樣本。
    合併兩組統計 (mean, var, n) 與 (batch_mean, batch_var, m)：
        δ     = batch_mean - mean
        mean' = mean + δ · m / (n + m)
        M2'   = var·n + batch_var·m + δ² · n·m / (n + m)
        var'  = M2' / (n + m)
    """

    def __init__(self, shape=(), eps=1e-8, device="cpu"):
        self.shape = tuple(shape)
        self.eps = eps
        self.mean = torch.zeros(self.shape, dtype=torch.float64, device=device)
        self.var = torch.ones(self.shape, dtype=torch.float64, device=device)
        self.count = 0  # 還沒看過任何樣本；此時 normalize 等於 (x - 0) / 1

    @torch.no_grad()
    def update(self, x):
        x = x.detach().to(self.mean.device, torch.float64).reshape(-1, *self.shape)
        m = x.shape[0]
        if m == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        n = self.count
        total = n + m
        delta = batch_mean - self.mean
        m2 = self.var * n + batch_var * m + delta ** 2 * n * m / total
        self.mean = self.mean + delta * m / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        """(x - mean) / sqrt(var + eps)"""
        mean, var = self.mean.to(x.dtype), self.var.to(x.dtype)
        return (x - mean) / torch.sqrt(var + self.eps)

    def denormalize(self, x):
        """normalize 的反函數：x · sqrt(var + eps) + mean"""
        mean, var = self.mean.to(x.dtype), self.var.to(x.dtype)
        return x * torch.sqrt(var + self.eps) + mean


# ============================================================================
# 3) ActorCritic
# ============================================================================
def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    """orthogonal 初始化（CleanRL 的 layer_init）。"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """預設的 Gaussian MLP policy，和 CleanRL ppo_continuous_action 同架構：
    actor / critic 各一個 64×2 tanh MLP，log_std 是和 obs 無關的可學習參數。
    動作不做 squash，由環境端 clip（和 CleanRL 一樣）。

    要換成別的 policy（tanh-Gaussian、GNN、加 agent ID 等）：繼承或重寫，
    只要實作 act / evaluate / value 三個方法即可，ppo_update 只用到它們。
    所有方法都接受 [..., obs_dim] 的任意前導維度。
    """

    def __init__(self, obs_dim, act_dim, critic_obs_dim=None, hidden=64):
        super().__init__()
        critic_obs_dim = obs_dim if critic_obs_dim is None else critic_obs_dim
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),  # 初始動作接近 0
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))

    def _dist(self, obs):
        mean = self.actor_mean(obs)
        return torch.distributions.Normal(mean, self.actor_logstd.exp().expand_as(mean))

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """rollout 用，不需要梯度。deterministic=True 時回傳 Gaussian 的 mean。
        回傳 (action [..., act_dim], logprob [...])；logprob 是各動作維度的總和。"""
        dist = self._dist(obs)
        action = dist.mean if deterministic else dist.sample()
        return action, dist.log_prob(action).sum(-1)

    def evaluate(self, obs, action):
        """回傳 (logprob [...], entropy [...])，各動作維度加總，保留梯度。"""
        dist = self._dist(obs)
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1)

    def value(self, critic_obs):
        """critic 吃 critic_obs（不是 obs）。回傳 [...]。"""
        return self.critic(critic_obs).squeeze(-1)


# ============================================================================
# 4) RolloutBuffer
# ============================================================================
class RolloutBuffer:
    """存放一次 rollout。每個欄位的前兩個以上維度是 [T, *B]：

        obs [T,*B,obs_dim]  critic_obs [T,*B,critic_obs_dim]  actions [T,*B,act_dim]
        logprobs, rewards, values, terminated, truncated, next_values, active : [T,*B]
        advantages, returns : [T,*B]（compute_advantages 之後才有值）

    單機時 critic_obs = obs（呼叫 add 時傳同一個 tensor 即可）。
    active 預設全為 1；補齊到 N_max 的位置請設為 0，之後所有 loss 與 advantage 正規化都會忽略它。
    advantages 保留的是「未正規化」的版本，供外部分析。
    """

    def __init__(self, T, batch_shape, obs_dim, act_dim, critic_obs_dim=None, device="cpu"):
        critic_obs_dim = obs_dim if critic_obs_dim is None else critic_obs_dim
        B = tuple(batch_shape)
        self.T, self.batch_shape, self.device = T, B, device

        def zeros(*extra):
            return torch.zeros((T, *B, *extra), device=device)

        self.obs = zeros(obs_dim)
        self.critic_obs = zeros(critic_obs_dim)
        self.actions = zeros(act_dim)
        self.logprobs = zeros()
        self.rewards = zeros()
        self.values = zeros()
        self.terminated = zeros()
        self.truncated = zeros()
        self.next_values = zeros()
        self.active = torch.ones((T, *B), device=device)
        self.advantages = zeros()
        self.returns = zeros()

    @torch.no_grad()  # buffer 裡的東西一律不帶梯度
    def add(self, t, obs, critic_obs, action, logprob, reward, value,
            terminated, truncated, next_value, active=None):
        """寫入第 t 步。每個引數的形狀是 [*B, ...]。"""
        self.obs[t] = obs
        self.critic_obs[t] = critic_obs
        self.actions[t] = action
        self.logprobs[t] = logprob
        self.rewards[t] = reward
        self.values[t] = value
        self.terminated[t] = terminated
        self.truncated[t] = truncated
        self.next_values[t] = next_value
        if active is not None:
            self.active[t] = active

    def compute_advantages(self, gamma, lam):
        self.advantages, self.returns = compute_gae(
            self.rewards, self.values, self.next_values,
            self.terminated, self.truncated, gamma, lam)


# ============================================================================
# 5) PPO 更新
# ============================================================================
@dataclass
class PPOConfig:
    # 預設值取自 CleanRL ppo_continuous_action
    n_epochs: int = 10
    n_minibatches: int = 32
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    clip_vloss: bool = True
    norm_adv: bool = True
    # advantage 正規化的範圍："minibatch"（CleanRL 做法，預設）或 "buffer"。
    # active 很稀疏時（例如補齊到 N_max 後大部分位置被遮掉），單一 minibatch 內的有效樣本
    # 可能很少甚至為 0，統計量很吵；"buffer" 在整個 buffer 上依 mask 正規化一次，比較穩。
    adv_norm_scope: str = "minibatch"
    # 動作均值越界懲罰（預設 0 = 關閉，行為與舊版逐位元相同）：loss += coef · E[relu(|μ|-1)²]。
    # μ 是 actor 的動作均值（pre-clip）。需要 agent.evaluate 之後把 μ 放在 agent.last_mean（沒有則報錯）。
    mean_bound_coef: float = 0.0


def masked_mean(x, mask):
    """依 mask 加權的平均：Σ(mask·x) / Σ(mask)。mask 必須是 0/1。全為 0 時回傳 0。
    用 torch.where 而不是乘法，所以 mask=0 位置的 NaN / inf 不會污染結果（NaN·0 = NaN）。"""
    return torch.where(mask > 0, x, torch.zeros_like(x)).sum() / mask.sum().clamp(min=1.0)


def masked_var(x, mask):
    """依 mask 加權的變異數（母體變異數）。mask 必須是 0/1。"""
    mean = masked_mean(x, mask)
    return masked_mean((x - mean) ** 2, mask)


def ppo_loss(agent, batch, cfg, value_rms=None):
    """單一 minibatch 的 PPO loss。batch 是攤平後的 dict：
    obs, critic_obs, actions, logprobs, values, advantages, returns, active（第 0 維都是樣本）。

    返回 dict：loss（可 backward）以及 policy_loss / value_loss / entropy / approx_kl / clip_frac。
    value_rms 不為 None 時，critic 輸出的是「正規化後的 value」，
    values、returns 會先用 value_rms.normalize 轉到同一個尺度再算 value loss。
    """
    mask = batch["active"]
    new_logprob, entropy = agent.evaluate(batch["obs"], batch["actions"])
    new_value = agent.value(batch["critic_obs"])

    # --- advantage 正規化（依 mask 加權的平均數與標準差）---
    adv = batch["advantages"]
    if cfg.norm_adv and cfg.adv_norm_scope == "minibatch":
        adv = (adv - masked_mean(adv, mask)) / (masked_var(adv, mask).sqrt() + 1e-8)

    # --- policy loss：clipped surrogate ---
    #   ratio = π_new(a|s) / π_old(a|s)
    #   L_pg  = E[ max( -A·ratio,  -A·clip(ratio, 1-ε, 1+ε) ) ]
    logratio = new_logprob - batch["logprobs"]
    ratio = logratio.exp()
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
    policy_loss = masked_mean(torch.max(pg_loss1, pg_loss2), mask)

    # --- value loss ---
    old_value, target = batch["values"], batch["returns"]
    if value_rms is not None:
        old_value, target = value_rms.normalize(old_value), value_rms.normalize(target)
    v_loss = (new_value - target) ** 2
    if cfg.clip_vloss:
        #   V_clip = V_old + clip(V_new - V_old, -ε, ε)
        #   L_v    = 0.5 · E[ max( (V_new - R)², (V_clip - R)² ) ]
        v_clipped = old_value + torch.clamp(new_value - old_value, -cfg.clip_coef, cfg.clip_coef)
        v_loss = torch.max(v_loss, (v_clipped - target) ** 2)
    value_loss = 0.5 * masked_mean(v_loss, mask)

    # --- entropy bonus ---
    entropy_mean = masked_mean(entropy, mask)

    # 總 loss = L_pg + c_v · L_v - c_ent · H
    loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_mean

    extra = {}
    if cfg.mean_bound_coef > 0:
        mu = getattr(agent, "last_mean", None)
        assert mu is not None, "mean_bound_coef > 0 需要 agent.evaluate 設定 agent.last_mean"
        over = torch.relu(mu.abs() - 1.0)
        mean_pen = masked_mean((over ** 2).mean(-1), mask)
        loss = loss + cfg.mean_bound_coef * mean_pen
        with torch.no_grad():
            extra = {"mean_pen": mean_pen.detach(), "mean_oob": masked_mean((mu.abs() > 1.0).float().mean(-1), mask)}

    with torch.no_grad():
        # k3 估計：KL(old || new) ≈ E[(ratio - 1) - log ratio]
        approx_kl = masked_mean((ratio - 1) - logratio, mask)
        clip_frac = masked_mean(((ratio - 1).abs() > cfg.clip_coef).float(), mask)
    return {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
            "entropy": entropy_mean, "approx_kl": approx_kl, "clip_frac": clip_frac, **extra}


def explained_variance(values, returns, mask):
    """1 - Var(R - V) / Var(R)，依 mask 加權。1 代表 critic 完美，0 代表和常數預測一樣差。"""
    var_ret = masked_var(returns, mask)
    if var_ret < 1e-12:
        return float("nan")
    return float(1.0 - masked_var(returns - values, mask) / var_ret)


def ppo_update(agent, optimizer, buffer, cfg, value_rms=None):
    """對 buffer 做 n_epochs × n_minibatches 次 PPO 更新，回傳統計量 dict。

    呼叫前 buffer 要已經填滿，且做過 buffer.compute_advantages(gamma, lam)。
    value_rms（可選）的呼叫端合約：
      - buffer.values 和 next_values 是「真實尺度」（GAE 在真實尺度上算），
        也就是呼叫端要把 critic 輸出用 value_rms.denormalize 轉回來再存。
      - critic 的輸出是「正規化尺度」。
      - value_rms 在本函式開頭用 active 的 returns 刷新，之後 old_value 與 returns 都用「新的」
        統計重新正規化。因此 value clip 的錨點（old_value）相對於 rollout 當時的 critic 輸出
        會有些微偏移，也沒有 PopArt 那樣的權重補償。這是已知限制。

    回傳（每個值是所有 minibatch 更新的平均）：
      policy_loss, value_loss, entropy, approx_kl, clip_frac, grad_norm（clip 之前的總梯度範數），
      explained_variance（更新前、整個 buffer，依 mask 加權）。
    """
    lead = buffer.rewards.shape  # [T, *B]
    N = buffer.rewards.numel()

    def flat(x):  # [T, *B, ...] -> [N, ...]
        return x.reshape(N, *x.shape[len(lead):])

    data = {k: flat(getattr(buffer, k)) for k in
            ["obs", "critic_obs", "actions", "logprobs", "values", "advantages", "returns", "active"]}

    if cfg.norm_adv and cfg.adv_norm_scope == "buffer":
        adv, m = data["advantages"], data["active"]   # buffer.advantages 本身不動
        data["advantages"] = (adv - masked_mean(adv, m)) / (masked_var(adv, m).sqrt() + 1e-8)
    elif cfg.adv_norm_scope not in ("minibatch", "buffer"):
        raise ValueError(f"adv_norm_scope 必須是 'minibatch' 或 'buffer'，收到 {cfg.adv_norm_scope!r}")

    if value_rms is not None:
        active = data["active"] > 0   # 只用有效元素更新統計，padding 不能影響 value_rms
        value_rms.update(data["returns"][active])

    ev = explained_variance(data["values"], data["returns"], data["active"])

    sums = {k: 0.0 for k in
            ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm"]}
    n_updates = 0
    for _ in range(cfg.n_epochs):
        perm = torch.randperm(N, device=buffer.device)
        for idx in torch.tensor_split(perm, cfg.n_minibatches):  # 最後幾個 minibatch 可能少一筆
            if len(idx) == 0:
                continue
            batch = {k: v[idx] for k, v in data.items()}
            out = ppo_loss(agent, batch, cfg, value_rms)

            optimizer.zero_grad()
            out["loss"].backward()
            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]:
                sums[k] += out[k].item()
            sums["grad_norm"] += float(grad_norm)
            for k in ("mean_pen", "mean_oob"):             # 只有 mean_bound_coef > 0 才有
                if k in out:
                    sums[k] = sums.get(k, 0.0) + out[k].item()
            n_updates += 1

    stats = {k: v / n_updates for k, v in sums.items()}
    stats["explained_variance"] = ev
    return stats
