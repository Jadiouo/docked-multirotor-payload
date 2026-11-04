"""整隊樣本的 PPO（rl_train --team-sample）。一筆樣本 = 一個環境（一整隊），不是一個模組。

為什麼放在這裡而不改 ppo.py：ppo.py 被其他專案複製使用，這裡只重用它的 masked_mean / explained_variance / compute_gae
等純函式，舊呼叫路徑完全不碰。

與 ppo.ppo_loss 的差別：
  - log-prob = 該隊所有有效模組（active=1）log-prob 的加總；ratio = exp(Δ加總)，每隊 clip 一次。
  - value 每隊一個（有效模組 value 的平均），GAE 在 [T, E] 上算。
  - entropy 取「加總」：聯合分佈（各模組在給定整隊 obs 下條件獨立）的熵就是各模組熵之和，
    與 log-prob 用加總一致；ent_coef 的尺度因此隨 N 成長（N 倍），若要沿用舊 ent_coef 需除以 N。目前 ent_coef=0。
  - 動作均值懲罰：對每隊的有效模組取平均，再對隊取平均（與舊版「對有效模組取平均」同尺度）。
buffer 的 batch_shape 是 (E,)，obs / critic_obs 只存每隊一列（模組 0 那列，整隊資料都在裡面），actions 為 [E, n_max·act_dim]。
"""
import torch
import torch.nn as nn

from ppo import masked_mean, masked_var, explained_variance


def team_ppo_loss(agent, batch, cfg, value_rms=None):
    mask = batch["active"]                                     # [M]，每隊 1（保留欄位，介面同 ppo_loss）
    new_logprob, entropy, mu, mod_active = agent.team_logprob_entropy(batch["obs"], batch["actions"])
    new_value = agent.team_value(batch["critic_obs"])

    adv = batch["advantages"]
    if cfg.norm_adv and cfg.adv_norm_scope == "minibatch":
        adv = (adv - masked_mean(adv, mask)) / (masked_var(adv, mask).sqrt() + 1e-8)

    logratio = new_logprob - batch["logprobs"]
    ratio = logratio.exp()
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
    policy_loss = masked_mean(torch.max(pg_loss1, pg_loss2), mask)

    old_value, target = batch["values"], batch["returns"]
    if value_rms is not None:
        old_value, target = value_rms.normalize(old_value), value_rms.normalize(target)
    v_loss = (new_value - target) ** 2
    if cfg.clip_vloss:
        v_clipped = old_value + torch.clamp(new_value - old_value, -cfg.clip_coef, cfg.clip_coef)
        v_loss = torch.max(v_loss, (v_clipped - target) ** 2)
    value_loss = 0.5 * masked_mean(v_loss, mask)

    entropy_mean = masked_mean(entropy, mask)
    loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_mean

    extra = {}
    if cfg.mean_bound_coef > 0:
        over = torch.relu(mu.abs() - 1.0)
        n_act = mod_active.sum(-1).clamp(min=1)
        per_team = lambda z: torch.where(mod_active > 0, z, torch.zeros_like(z)).sum(-1) / n_act
        mean_pen = masked_mean(per_team((over ** 2).mean(-1)), mask)
        loss = loss + cfg.mean_bound_coef * mean_pen
        with torch.no_grad():
            extra = {"mean_pen": mean_pen.detach(), "mean_oob": masked_mean(per_team((mu.abs() > 1.0).float().mean(-1)), mask)}

    with torch.no_grad():
        approx_kl = masked_mean((ratio - 1) - logratio, mask)
        clip_frac = masked_mean(((ratio - 1).abs() > cfg.clip_coef).float(), mask)
    return {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
            "entropy": entropy_mean, "approx_kl": approx_kl, "clip_frac": clip_frac, **extra}


def team_ppo_update(agent, optimizer, buffer, cfg, value_rms=None):
    """同 ppo.ppo_update，但樣本是「隊」：buffer 形狀 [T, E, ...]，minibatch 以隊為單位切。"""
    N = buffer.rewards.numel()
    flat = lambda x: x.reshape(N, *x.shape[buffer.rewards.dim():])
    data = {k: flat(getattr(buffer, k)) for k in
            ["obs", "critic_obs", "actions", "logprobs", "values", "advantages", "returns", "active"]}
    if cfg.norm_adv and cfg.adv_norm_scope == "buffer":
        adv, m = data["advantages"], data["active"]
        data["advantages"] = (adv - masked_mean(adv, m)) / (masked_var(adv, m).sqrt() + 1e-8)
    elif cfg.adv_norm_scope not in ("minibatch", "buffer"):
        raise ValueError(f"adv_norm_scope 必須是 'minibatch' 或 'buffer'，收到 {cfg.adv_norm_scope!r}")
    if value_rms is not None:
        value_rms.update(data["returns"][data["active"] > 0])
    ev = explained_variance(data["values"], data["returns"], data["active"])

    sums = {k: 0.0 for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm"]}
    n_updates = 0
    for _ in range(cfg.n_epochs):
        perm = torch.randperm(N, device=buffer.device)
        for idx in torch.tensor_split(perm, cfg.n_minibatches):
            if len(idx) == 0:
                continue
            out = team_ppo_loss(agent, {k: v[idx] for k, v in data.items()}, cfg, value_rms)
            optimizer.zero_grad()
            out["loss"].backward()
            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()
            for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]:
                sums[k] += out[k].item()
            sums["grad_norm"] += float(grad_norm)
            for k in ("mean_pen", "mean_oob"):
                if k in out:
                    sums[k] = sums.get(k, 0.0) + out[k].item()
            n_updates += 1
    stats = {k: v / n_updates for k, v in sums.items()}
    stats["explained_variance"] = ev
    return stats
