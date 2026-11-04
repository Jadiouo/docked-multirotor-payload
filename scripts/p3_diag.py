"""P3 診斷：在指定任務上跑 checkpoint（deterministic 與隨機各一次），拆解 reward 各項、終止原因、動作飽和比例。
只讀 checkpoint 與 env，不改任何東西。用法（CPU 即可）：
  CUDA_VISIBLE_DEVICES= .venv-gpu/bin/python -m scripts.p3_diag results/p3/single_T1/final.pt --config single --task T2
"""
import argparse, json, math
import torch
from quadsim.env import ModularEnv
from quadsim.rl_env import PackedEnv
from scripts.rl_eval import load_policy
from scripts.rl_train import ALL_CONFIGS


def run(agent, cfg, task, n, det, seed=123):
    env = ModularEnv(ALL_CONFIGS[cfg], n, task=task, seed=seed)
    pe = PackedEnv(env, agent.n_max, agent.hist)
    obs, _ = pe.reset()
    c = env.cfg
    acc = dict(pos=0., vel=0., rate=0., yaw=0., act=0., reward=0.); sat = 0.; cnt = 0
    alive = torch.ones(n, dtype=torch.bool); term_why = dict(ground=0, far=0, tilt=0, trunc=0)
    pos2 = 0.
    for s in range(c.episode_steps):
        with torch.no_grad():
            a, _ = agent.act(obs, deterministic=det)
        obs, _, r, te, tr, info = pe.step(a)
        fs = info["final_state"]
        m = alive.float()
        e_p = (fs["goal_pos"] - fs["pos"]).norm(dim=-1)
        acc["pos"] += (e_p * m).sum().item(); acc["vel"] += (fs["vel"].norm(dim=-1) * m).sum().item()
        acc["rate"] += (fs["w"].norm(dim=-1) * m).sum().item(); acc["reward"] += (r * m).sum().item()
        pos2 += ((e_p ** 2) * m).sum().item()
        sat += ((a[:, :env.N].abs() >= 1.0).float().mean((1, 2)) * m).sum().item(); cnt += m.sum().item()
        # 終止原因（在 alive 的環境上，te 發生的那一步）
        new_t = te & alive
        if new_t.any():
            R = fs["pos"][:, 2] < 0
            far = e_p > c.max_pos_err
            term_why["ground"] += int((new_t & R).sum()); term_why["far"] += int((new_t & far & ~R).sum())
            term_why["tilt"] += int((new_t & ~R & ~far).sum())
        alive = alive & ~(te | tr)
    term_why["trunc"] = int(alive.sum())
    return dict(mean_pos_err=acc["pos"] / cnt, rms_pos=math.sqrt(pos2 / cnt), mean_speed=acc["vel"] / cnt,
                mean_rate=acc["rate"] / cnt, step_reward=acc["reward"] / cnt, action_saturated_frac=sat / cnt,
                steps_alive=cnt / n, term=term_why)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--config", default="single")
    ap.add_argument("--task", default="T2"); ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    agent, ck = load_policy(a.ckpt)
    out = {}
    for det in (True, False):
        out["det" if det else "stoch"] = run(agent, a.config, a.task, a.n, det)
    out["log_std"] = agent.actor_logstd.detach().tolist()
    print(json.dumps(out, indent=1))
