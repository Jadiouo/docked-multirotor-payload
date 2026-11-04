"""用 ppo.py 訓練 ModularActorCritic（MAPPO 風格：參數共享的 actor ＋ 集中式 critic）。

多構型：每個構型（可寫成 name:TASK 指定任務）各自一組批次環境，rollout 時各自 step，
補齊到 N_max（見 quadsim/rl_env.py），所有構型的資料合成同一個 buffer，一次 ppo_update。
reward 直接用 env 的 reward（整隊共用，expand 成 [E, N_max]）；補齊的模組靠 active mask 排除。
final_obs 照 ppo.py 的約定：next_values 用 reset 之前的 final obs 算，terminated 的那步再由 GAE 遮掉。
value 用 RunningMeanStd 正規化（critic 輸出正規化尺度，buffer 存真實尺度，合約見 ppo_update 說明）。

輸出到 results/p3/<run_name>/：args.json、train.csv（每次更新一列）、latest.pt（最新）、final.pt。
執行（專案根目錄）：
    .venv-gpu/bin/python -m scripts.rl_train --run-name t2_single --configs single:T2 --n-envs 2048 \
    --total-steps 40000000 --device cuda
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from ppo import PPOConfig, RolloutBuffer, RunningMeanStd, ppo_update
from quadsim.assembly import CONFIGS
from quadsim.env import ModularEnv, TASKS
from quadsim.policy import ModularActorCritic
from quadsim.team_ppo import team_ppo_update
from quadsim.rl_env import PackedEnv, RandomPayloadEnv, ResidualEnv, make_estim_yawlim_controller
from quadsim.policy import OBS_DIM
from scripts.tune_classical import TRAIN_CONFIGS

ALL_CONFIGS = {**CONFIGS, **TRAIN_CONFIGS}
RES = Path(__file__).resolve().parent.parent / "results" / "p3"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--configs", nargs="+", default=["single"], help="構型名稱，可寫成 name:TASK（例如 single:T2）")
    ap.add_argument("--task", default="T2", choices=TASKS, help="沒有寫 :TASK 的構型用這個任務")
    ap.add_argument("--n-envs", type=int, nargs="+", default=[1024], help="每個構型的環境數；給一個值就全部共用")
    ap.add_argument("--rollout-T", type=int, default=32, help="每次更新前每個環境走幾步")
    ap.add_argument("--total-steps", type=int, default=20_000_000, help="總環境步數（所有環境、所有構型加總）")
    ap.add_argument("--time-limit-min", type=float, default=None, help="超過就提早結束（存檔後離開）")
    ap.add_argument("--k", type=int, default=1, help="message passing 輪數；0 = 只看自己")
    ap.add_argument("--hist", type=int, default=1, help="觀測歷史長度 H；1 = 沒有記憶")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--no-layer-norm", action="store_true")
    ap.add_argument("--log-std-init", type=float, default=-1.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-anneal-lr", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--n-minibatches", type=int, default=16)
    ap.add_argument("--clip-coef", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--adv-norm-scope", default="buffer", choices=["minibatch", "buffer"])
    ap.add_argument("--payload-max", type=float, default=0.0,
                    help="未知 payload：每個環境每次 reset 抽質量比例 U(0, pmax)，掛點模組在所有模組中均勻抽（偏移 0，同 g2 做法；g2 評估是固定 15%、argmax x，不可當訓練分布）；0 = 關閉（行為與舊版完全相同）。"
                         "actor 看不到 payload；critic 的特權欄位 d 是含 payload 的真實 CoM 相對位置")
    ap.add_argument("--payload-offset-max", type=float, default=0.0,
                    help="payload 掛點偏離模組中心：比例 s（需要 --payload-max > 0）。x、y 各 U(−s·0.0333, s·0.0333) m，z U(−s·0.0215, 0) m"
                         "（0.0333、0.0215 = P3d 偏離中心診斷的固定偏移 (+0.0333, −0.0333, −0.0215) 各分量的絕對值，由 0.718·0.4·pitch、0.718·0.03 算出）；0 = 關閉（偏移 0，行為與舊版相同）")
    ap.add_argument("--payload-event-prob", type=float, default=0.0,
                    help="每個 episode 以此機率改成中途 payload 事件（attach／drop 各半，事件時刻 U(1.0, 3.5) s；需要 --payload-max > 0）；0 = 關閉（行為與舊版相同）。"
                         "attach：事件前無 payload，之後有；drop：相反。沒被選到的 episode 整段帶 payload。")
    ap.add_argument("--mean-bound-coef", type=float, default=0.0,
                    help="動作均值越界懲罰係數：loss += coef·mean(relu(|μ|-1)²)，μ 為 pre-clip 均值；0 = 關閉（行為與舊版逐位元相同）。train.csv 另記 mean_pen、mean_oob（|μ|>1 比例）")
    ap.add_argument("--team-sample", action="store_true",
                    help="PPO 樣本單位改成整隊（每個環境一筆）：log-prob＝有效模組加總、每隊 clip 一次、value 每隊一個、GAE 在 [E] 上算、minibatch 以隊切；"
                         "entropy 取加總。關閉時行為與舊版逐位元相同。注意樣本數變成 1/N，n_minibatches 要檢查（每個 minibatch 至少幾千隊）")
    ap.add_argument("--residual", action="store_true",
                    help="殘差 RL（探索性，prereg 第 13 節）：u = clip(u_base + scale·δ, −1, 1)，u_base = B1-QP-estim-yawlim（snap=False）的指令，policy 輸出 δ；"
                         "obs 多接 u_base（每模組 +4 維）。初始 δ≈0（最後一層權重縮小、log_std 調小），初始策略 = 古典法。關閉時行為與舊版逐位元相同。"
                         "控制器在 CPU 上逐步求解，比純 RL 慢")
    ap.add_argument("--residual-scale", type=float, default=0.2, help="--residual 的 scale")
    ap.add_argument("--residual-log-std-init", type=float, default=-2.0, help="--residual 時 δ 的 log_std 初始值（取代 --log-std-init）")
    ap.add_argument("--residual-head-std", type=float, default=1e-3, help="--residual 時 actor 最後一層權重的初始 std（舊預設 0.01，縮小讓 δ 初始更接近 0）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=20, help="每幾次更新存一次 latest.pt")
    return ap.parse_args()


def make_groups(args, device):
    """每個 (構型, 任務) 一組批次環境。回傳 [(名稱, PackedEnv)]。"""
    n_envs = args.n_envs * len(args.configs) if len(args.n_envs) == 1 else args.n_envs
    assert len(n_envs) == len(args.configs), "--n-envs 的個數要等於構型個數（或只給一個）"
    specs = [(c.split(":") + [args.task])[:2] for c in args.configs]
    n_max = max(len(ALL_CONFIGS[name]) for name, _ in specs)
    groups = []
    for i, ((name, task), n) in enumerate(zip(specs, n_envs)):
        if args.payload_max > 0:
            env = RandomPayloadEnv(ALL_CONFIGS[name], n, args.payload_max, task=task, device=device, seed=args.seed * 1000 + i,
                                   offset_max=args.payload_offset_max, event_prob=args.payload_event_prob)
        else:
            assert args.payload_offset_max == 0 and args.payload_event_prob == 0, "--payload-offset-max／--payload-event-prob 需要 --payload-max > 0"
            env = ModularEnv(ALL_CONFIGS[name], n, task=task, device=device, seed=args.seed * 1000 + i)
        if args.residual:
            env = ResidualEnv(env, make_estim_yawlim_controller(), args.residual_scale)
        groups.append((f"{name}:{task}", PackedEnv(env, n_max, args.hist)))
    return groups, n_max


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device
    if device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.6)
        # 注意：不開 TF32。env 的推力→力矩用 float32 matmul，TF32 的精度（10 bit 尾數）會破壞 hover 時的力矩抵消。
    out = RES / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    groups, n_max = make_groups(args, device)
    names = [g[0] for g in groups]
    sizes = [pe.B for _, pe in groups]
    E = sum(sizes)
    gid = torch.cat([torch.full((s,), i, device=device) for i, s in enumerate(sizes)])        # 每個環境屬於哪組
    T = args.rollout_T

    # hover 動作（各構型相同，因為推重比與 N 無關）：env reset 時 u_prev 就設在 hover
    hover_action = float(groups[0][1].env.u_prev[0, 0, 0])
    if args.residual:       # δ 的 mean 以 0 為中心（不是 hover），obs 多 4 維 u_base
        akw = dict(obs_dim=OBS_DIM + 4, log_std_init=args.residual_log_std_init, hover_action=0.0, head_std=args.residual_head_std)
    else:
        akw = dict(log_std_init=args.log_std_init, hover_action=hover_action)
    agent = ModularActorCritic(n_max, hist=args.hist, hidden=args.hidden, k=args.k,
                               layer_norm=not args.no_layer_norm, **akw).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    cfg = PPOConfig(n_epochs=args.n_epochs, n_minibatches=args.n_minibatches, clip_coef=args.clip_coef,
                    ent_coef=args.ent_coef, adv_norm_scope=args.adv_norm_scope,
                    mean_bound_coef=args.mean_bound_coef)
    value_rms = RunningMeanStd(device=device)
    active = torch.cat([pe.active.expand(pe.B, n_max) for _, pe in groups])                   # [E, n_max]，不隨時間變
    team = args.team_sample
    if team:   # 一筆樣本 = 一隊：obs 只存一列（整隊資料都在裡面），actions 攤平成 [E, n_max·4]
        buf = RolloutBuffer(T, (E,), agent.actor_layout.size, n_max * 4, agent.critic_layout.size, device)
    else:
        buf = RolloutBuffer(T, (E, n_max), agent.actor_layout.size, 4, agent.critic_layout.size, device)

    def step_envs(action):
        """各組環境各自 step，結果串接回 [E, ...]。"""
        outs, lo = [], 0
        for _, pe in groups:
            outs.append(pe.step(action[lo:lo + pe.B])); lo += pe.B
        cat = lambda xs: torch.cat(xs, 0)
        obs, cobs, rew, term, trunc = (cat([o[i] for o in outs]) for i in range(5))
        f_obs, f_cobs = cat([o[5]["final_obs"] for o in outs]), cat([o[5]["final_critic_obs"] for o in outs])
        return obs, cobs, rew, term, trunc, f_obs, f_cobs

    def v_real(cobs):                     # critic 輸出（正規化尺度）→ 真實尺度
        # 用整隊共用的入口：一個環境只算一次 GNN（與逐列的 agent.value 數值相同，少 n_max 倍計算）
        v = value_rms.denormalize(agent.value_team(cobs))
        if team:                           # 每隊一個 value：有效模組的平均（假模組的 value 為 0，不算）
            return (v * active).sum(1) / active.sum(1).clamp(min=1)
        return v

    n_iters = args.total_steps // (E * T)
    assert n_iters > 0, f"total-steps 至少要 {E * T}"
    G = len(groups)
    ep_ret, ep_len = torch.zeros(E, device=device), torch.zeros(E, device=device)
    obs, cobs = (torch.cat(x, 0) for x in zip(*[pe.reset() for _, pe in groups]))
    with torch.no_grad():
        value = v_real(cobs)

    fields = ["iter", "global_step", "module_steps", "time_s", "step_reward", "ep_return", "ep_len", "success_rate"]
    fields += [f"{k}/{n}" for n in names for k in ("ep_return", "success_rate")] if G > 1 else []
    fields += ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm", "explained_variance", "lr", "log_std", "mean_pen", "mean_oob"]
    f = open(out / "train.csv", "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
    mods_per_iter = sum(pe.N * pe.B for _, pe in groups) * T          # 每次更新的 module-steps

    def save(path, it):
        torch.save(dict(model=agent.state_dict(), hp=agent.hp, value_rms=dict(
            mean=value_rms.mean, var=value_rms.var, count=value_rms.count), iter=it, args=vars(args)), path)

    start = time.time()
    for it in range(1, n_iters + 1):
        lr = args.lr if args.no_anneal_lr else args.lr * (1.0 - (it - 1.0) / n_iters)
        optimizer.param_groups[0]["lr"] = lr
        n_done, s_ret, s_len, n_succ = (torch.zeros(G, device=device) for _ in range(4))
        sum_rew = torch.zeros((), device=device)

        for t in range(T):
            action, logprob = agent.act_team(obs)           # 同 agent.act，但整隊只算一次 GNN
            n_obs, n_cobs, reward, term, trunc, f_obs, f_cobs = step_envs(action)
            done = term | trunc
            with torch.no_grad():
                v_final = v_real(f_cobs)                                       # V(s_{t+1})，done 的環境用 reset 之前的 final obs
                v_new = v_real(n_cobs)                                         # 下一步的 V(s)，done 的環境是新 episode
            next_value = v_final
            if team:
                lp = torch.where(active > 0, logprob, torch.zeros_like(logprob)).sum(1)   # 隊 log-prob = 有效模組加總
                buf.add(t, obs[:, 0], cobs[:, 0], action.reshape(E, -1), lp, reward, value, term.float(), trunc.float(),
                        next_value)
            else:
                e = lambda x: x[:, None].expand(E, n_max)                      # 整隊共用的量 → 每個模組一份
                buf.add(t, obs, cobs, action, logprob, e(reward), value, e(term.float()), e(trunc.float()),
                        next_value, active)
            obs, cobs = n_obs, n_cobs
            value = torch.where(done if team else done[:, None], v_new, v_final)

            ep_ret += reward; ep_len += 1; sum_rew += reward.mean()
            fin = done.float()
            n_done.index_add_(0, gid, fin); s_ret.index_add_(0, gid, ep_ret * fin)
            s_len.index_add_(0, gid, ep_len * fin); n_succ.index_add_(0, gid, trunc.float())
            ep_ret *= 1 - fin; ep_len *= 1 - fin

        buf.compute_advantages(args.gamma, args.gae_lambda)
        stats = (team_ppo_update if team else ppo_update)(agent, optimizer, buf, cfg, value_rms)

        # ---- 記錄（每次更新一列；只在這裡同步 GPU）----
        nd = n_done.cpu().numpy(); sr, sl, ns = s_ret.cpu().numpy(), s_len.cpu().numpy(), n_succ.cpu().numpy()
        div = lambda a, b: a / b if b > 0 else float("nan")
        gs = it * E * T
        row = dict(iter=it, global_step=gs, module_steps=mods_per_iter * it, time_s=round(time.time() - start, 1),
                   step_reward=float(sum_rew) / T, ep_return=div(sr.sum(), nd.sum()), ep_len=div(sl.sum(), nd.sum()),
                   success_rate=div(ns.sum(), nd.sum()), lr=lr, log_std=float(agent.actor_logstd.detach().mean()),
                   **{k: stats[k] for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
                                            "grad_norm", "explained_variance"]},
                   mean_pen=stats.get("mean_pen", 0.0), mean_oob=stats.get("mean_oob", 0.0))
        if G > 1:
            for i, n in enumerate(names):
                row[f"ep_return/{n}"], row[f"success_rate/{n}"] = div(sr[i], nd[i]), div(ns[i], nd[i])
        writer.writerow(row); f.flush()
        finite = all(math.isfinite(row[k]) for k in ["policy_loss", "value_loss", "entropy"])
        print(f"[{it}/{n_iters}] t={row['time_s']:.0f}s step_r={row['step_reward']:.3f} ep_ret={row['ep_return']:.1f} "
              f"len={row['ep_len']:.0f} succ={row['success_rate']:.2f} kl={row['approx_kl']:.4f} "
              f"ev={row['explained_variance']:.2f} H={row['entropy']:.2f}", flush=True)
        if not finite:
            print("loss 不是有限值，停止", flush=True); break
        if it % args.save_every == 0:
            save(out / "latest.pt", it)
        if args.time_limit_min and (time.time() - start) / 60 > args.time_limit_min:
            print("達到時間上限，提早結束", flush=True); break
    save(out / "final.pt", it)
    f.close()
    print("saved", out / "final.pt")


if __name__ == "__main__":
    main()
