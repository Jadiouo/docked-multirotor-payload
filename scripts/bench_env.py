"""驗收 5：env.step 的速度（module-steps/s = B × N × 步數 / 秒）。
執行（專案根目錄，CPU）：
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 .venv/bin/python -m scripts.bench_env [--B 1024] [--config square4]"""
import argparse
import time
import torch
from quadsim.assembly import CONFIGS
from quadsim.env import ModularEnv


def bench(config, B, steps, warmup, repeats=7, task="T2"):
    env = ModularEnv(CONFIGS[config], B, task, seed=0)
    env.reset()
    u = torch.rand(B, env.N, 4) * 2 - 1
    for _ in range(warmup): env.step(u)
    rates = []
    for _ in range(repeats):                              # 機器忙的時候單次量測很不穩，重複量取中位數與最大值
        t0 = time.perf_counter()
        for _ in range(steps): env.step(u)
        rates.append(B * env.N * steps / (time.perf_counter() - t0))
    rates.sort()
    return rates[len(rates) // 2], rates[-1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--config", default="square4")      # N = 4
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    print(f"torch threads = {torch.get_num_threads()}")
    for cfg in (a.config, "single", "grid6"):
        med, best = bench(cfg, a.B, a.steps, warmup=20)
        print(f"{cfg:8s} B={a.B}: 中位數 {med / 1e3:8.1f} k module-steps/s，最大 {best / 1e3:8.1f} k（每個 env step = 5 個物理子步）")
