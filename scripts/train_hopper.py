"""用 ppo.py 在 Hopper-v5 上訓練，與 CleanRL ppo_continuous_action 對照（P1 驗收）。

超參數與環境處理刻意與 third_party/cleanrl/ppo_continuous_action.py 完全一致：
1 env、2048 步、10 epochs、32 minibatches、lr 3e-4 線性衰減、gamma 0.99、gae 0.95、clip 0.2、
vf clip、ent 0、vf 0.5、max_grad_norm 0.5。

obs / reward 處理：直接用和 CleanRL 同一組 gymnasium wrapper（FlattenObservation、
RecordEpisodeStatistics、ClipAction、NormalizeObservation、clip ±10、NormalizeReward、clip ±10），
不用 train_pendulum.py 的手寫正規化。理由：wrapper 本身就是 CleanRL 的定義，用同一組才能保證
「差異只來自 PPO 實作」。wrapper 在 env 內層，所以 final_obs 也已經過正規化與 clip。

與 CleanRL 對齊的取捨：CleanRL 把 truncated（time limit）當成 done、不 bootstrap。預設照做
（truncated 併入 terminated）；加 --bootstrap-truncation 則改用 ppo.py 原生的 truncated 處理。

輸出：results/p1/hopper_ours_seed{S}.csv，欄位 global_step, episodic_return（每個 episode 一列，
global_step 為 episode 結束時的步數），格式與 CleanRL 版相同。
用法：python scripts/train_hopper.py --seed 1
"""
import argparse
import csv
import os
import random
import sys
import time

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ppo import ActorCritic, PPOConfig, RolloutBuffer, ppo_update  # noqa: E402

# ------------------------------ 超參數（= CleanRL 預設） ------------------------------
ENV_ID = "Hopper-v5"
NUM_ENVS = 1
NUM_STEPS = 2048
LR = 3e-4
ANNEAL_LR = True
GAMMA = 0.99
GAE_LAMBDA = 0.95
CFG = PPOConfig(n_epochs=10, n_minibatches=32, clip_coef=0.2, vf_coef=0.5,
                ent_coef=0.0, max_grad_norm=0.5, clip_vloss=True, norm_adv=True)
# ---------------------------------------------------------------------------------------


def make_env(env_id, gamma):
    """與 CleanRL make_env 逐行相同（gymnasium 1.x 的 TransformObservation 需要傳 observation_space）。"""
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--env-id", default=ENV_ID)
    parser.add_argument("--out-dir", default="results/p1")
    parser.add_argument("--bootstrap-truncation", action="store_true",
                        help="time limit 時 bootstrap（ppo.py 原生行為）；預設與 CleanRL 相同，不 bootstrap")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)

    # SAME_STEP = gymnasium 0.29 的 autoreset 語意（CleanRL 原本的假設）
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, GAMMA) for _ in range(NUM_ENVS)],
                                    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    agent = ActorCritic(obs_dim, act_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    buf = RolloutBuffer(NUM_STEPS, (NUM_ENVS,), obs_dim, act_dim)

    n_iters = args.total_timesteps // (NUM_ENVS * NUM_STEPS)
    assert n_iters > 0, f"total-timesteps 至少要 {NUM_ENVS * NUM_STEPS}"
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"hopper_ours_seed{args.seed}.csv")

    global_step = 0
    start = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(next_obs, dtype=torch.float32)
    with torch.no_grad():
        value = agent.value(obs)          # V(目前 obs)；每步只做一次 critic forward，下一步重用

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["global_step", "episodic_return"])
        for it in range(1, n_iters + 1):
            if ANNEAL_LR:
                optimizer.param_groups[0]["lr"] = LR * (1.0 - (it - 1.0) / n_iters)

            for t in range(NUM_STEPS):
                global_step += NUM_ENVS
                action, logprob = agent.act(obs)
                # 動作不在這裡 clip：ClipAction wrapper 在環境內做，buffer 存原始動作（同 CleanRL）
                next_obs, reward, terminated, truncated, infos = envs.step(action.numpy())
                next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32)
                done = terminated | truncated

                with torch.no_grad():
                    next_value = agent.value(next_obs_t)   # 結束時是新 episode 的 obs，被 terminated 遮掉
                    if args.bootstrap_truncation and truncated.any():
                        for i in np.flatnonzero(truncated & ~terminated):
                            fin = torch.as_tensor(infos["final_obs"][i], dtype=torch.float32)
                            next_value[i] = agent.value(fin)   # final_obs 已被 env 內層 wrapper 正規化
                        term_flag = terminated
                    else:
                        term_flag = done                        # CleanRL：truncated 當 terminated，不 bootstrap
                buf.add(t, obs, obs, action, logprob, torch.as_tensor(reward, dtype=torch.float32), value,
                        torch.as_tensor(term_flag, dtype=torch.float32),
                        torch.as_tensor(truncated & ~term_flag if args.bootstrap_truncation else np.zeros_like(truncated),
                                        dtype=torch.float32),
                        next_value)

                if "final_info" in infos and "episode" in infos["final_info"]:
                    fi = infos["final_info"]
                    for i in np.flatnonzero(infos["_final_info"] & fi["_episode"]):
                        ret = float(fi["episode"]["r"][i])
                        print(f"global_step={global_step}, episodic_return={ret}", flush=True)
                        writer.writerow([global_step, ret])
                        f.flush()

                obs, value = next_obs_t, next_value
                if args.bootstrap_truncation and truncated.any():
                    with torch.no_grad():
                        value = agent.value(next_obs_t)   # 下一步的 V(s) 要用新 episode 的 obs

            buf.compute_advantages(GAMMA, GAE_LAMBDA)
            ppo_update(agent, optimizer, buf, CFG)
            print("SPS:", int(global_step / (time.time() - start)), flush=True)
    envs.close()
    print("saved", csv_path)


if __name__ == "__main__":
    main()
