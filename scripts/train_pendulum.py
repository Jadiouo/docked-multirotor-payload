"""用 ppo.py 在 Pendulum-v1 上訓練，存學習曲線 CSV 到 results/p1/。

設定仿 CleanRL ppo_continuous_action（obs 與 reward 正規化、lr 線性衰減、
10 epochs × 32 minibatches），env 數與每次 rollout 長度為 Pendulum 調過。
用法：python scripts/train_pendulum.py --seed 1 --total-steps 500000
"""
import argparse
import csv
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ppo import ActorCritic, PPOConfig, RolloutBuffer, RunningMeanStd, ppo_update  # noqa: E402

# ------------------------------ 超參數 ------------------------------
ENV_ID = "Pendulum-v1"
NUM_ENVS = 4
NUM_STEPS = 512            # 每個 env 每次 rollout 的步數；batch = 4 × 512 = 2048
TOTAL_STEPS = 500_000      # 預設值，可用 --total-steps 覆蓋
LR = 3e-4
ANNEAL_LR = True
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_OBS = 10.0            # obs 與 reward 正規化後的 clip 範圍
CFG = PPOConfig(n_epochs=10, n_minibatches=32, clip_coef=0.2, vf_coef=0.5,
                ent_coef=0.0, max_grad_norm=0.5, clip_vloss=True, norm_adv=True)
# --------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--out-dir", default="results/p1")
    args = parser.parse_args()

    torch.set_num_threads(2)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # SAME_STEP：episode 結束的那一步直接回傳新 episode 的 obs，
    # 真正的最後一個 obs 放在 info["final_obs"]，bootstrap 要用它。
    envs = gym.make_vec(ENV_ID, num_envs=NUM_ENVS, vectorization_mode="sync",
                        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    low, high = envs.single_action_space.low, envs.single_action_space.high

    agent = ActorCritic(obs_dim, act_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    buf = RolloutBuffer(NUM_STEPS, (NUM_ENVS,), obs_dim, act_dim)
    obs_rms = RunningMeanStd((obs_dim,))
    ret_rms = RunningMeanStd(())   # 用「折扣回報累積值」的標準差縮放 reward（CleanRL NormalizeReward）
    ret_acc = np.zeros(NUM_ENVS)

    def norm_obs(x):
        x = obs_rms.normalize(torch.as_tensor(x, dtype=torch.float32))
        return x.clamp(-CLIP_OBS, CLIP_OBS)

    obs, _ = envs.reset(seed=args.seed)
    obs_rms.update(torch.as_tensor(obs))
    obs = norm_obs(obs)
    ep_ret = np.zeros(NUM_ENVS)       # 進行中 episode 的原始回報
    finished = []                     # 已完成 episode 的原始回報
    n_updates = args.total_steps // (NUM_STEPS * NUM_ENVS)
    assert n_updates > 0, f"total-steps 至少要 {NUM_STEPS * NUM_ENVS}（一次 rollout 的步數）"
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"pendulum_seed{args.seed}.csv")
    fields = ["update", "global_step", "mean_return_10", "policy_loss", "value_loss", "entropy",
              "approx_kl", "clip_frac", "explained_variance", "grad_norm", "sps"]
    start = time.time()
    global_step = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for update in range(1, n_updates + 1):
            if ANNEAL_LR:
                optimizer.param_groups[0]["lr"] = LR * (1.0 - (update - 1) / n_updates)

            for t in range(NUM_STEPS):
                global_step += NUM_ENVS
                action, logprob = agent.act(obs)
                value = agent.value(obs).detach()
                a_env = np.clip(action.numpy(), low, high)  # 環境端 clip；buffer 存未 clip 的動作
                next_obs, reward, terminated, truncated, info = envs.step(a_env)

                # 真正的 s_{t+1}：結束的 env 要換成 final_obs，其他 env 就是 next_obs
                real_next = np.array(next_obs, copy=True)
                done = terminated | truncated
                if done.any():
                    for i in np.flatnonzero(info["_final_obs"]):
                        real_next[i] = info["final_obs"][i]

                # 先用真正的 s_{t+1}（含 final_obs）更新 obs 統計，再 normalize：
                # 同一個 s_{t+1} 用於 next_value 與下一步的 obs，必須是同一份統計
                obs_rms.update(torch.as_tensor(real_next))

                # reward 縮放
                ep_ret += reward
                ret_acc = ret_acc * GAMMA + reward
                ret_rms.update(torch.as_tensor(ret_acc))
                scaled = np.clip(reward / np.sqrt(ret_rms.var.item() + 1e-8), -CLIP_OBS, CLIP_OBS)
                ret_acc[done] = 0.0

                # next_value 用 final obs 算（以目前的 obs 統計正規化）
                with torch.no_grad():
                    next_value = agent.value(norm_obs(real_next))
                buf.add(t, obs, obs, action, logprob, torch.as_tensor(scaled, dtype=torch.float32),
                        value, torch.as_tensor(terminated), torch.as_tensor(truncated), next_value)

                for i in np.flatnonzero(done):
                    finished.append(ep_ret[i])
                    ep_ret[i] = 0.0
                obs = norm_obs(next_obs)

            buf.compute_advantages(GAMMA, GAE_LAMBDA)
            stats = ppo_update(agent, optimizer, buf, CFG)

            row = {"update": update, "global_step": global_step,
                   "mean_return_10": float(np.mean(finished[-10:])) if finished else float("nan"),
                   "sps": int(global_step / (time.time() - start)), **stats}
            writer.writerow(row)
            f.flush()
            if update % 10 == 0 or update == n_updates:
                print(f"update {update}/{n_updates} step {global_step} "
                      f"return10 {row['mean_return_10']:.1f} kl {stats['approx_kl']:.4f} "
                      f"ev {stats['explained_variance']:.2f} sps {row['sps']}", flush=True)
    envs.close()
    print("saved", csv_path)


if __name__ == "__main__":
    main()
