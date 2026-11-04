"""環境在 GPU 上的速度（只量 env.step，不含 policy）：module-steps/s = B × N × 步數 / 秒。
同時印出 uptime 的 load（機器很忙時數字會偏低）。
執行：
    .venv-gpu/bin/python -m scripts.bench_env_gpu [--B 4096] [--device cuda]"""
import argparse
import subprocess
import time
import torch
from quadsim.assembly import CONFIGS
from quadsim.env import ModularEnv


def bench(config, B, device, steps=50, warmup=10, repeats=5):
    env = ModularEnv(CONFIGS[config], B, "T2", device=device, seed=0)
    env.reset()
    u = torch.rand(B, env.N, 4, device=device) * 2 - 1
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    for _ in range(warmup): env.step(u)
    rates = []
    for _ in range(repeats):
        sync(); t0 = time.perf_counter()
        for _ in range(steps): env.step(u)
        sync(); rates.append(B * env.N * steps / (time.perf_counter() - t0))
    rates.sort()
    return env.N, rates[len(rates) // 2], rates[-1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, nargs="+", default=[4096])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--configs", nargs="+", default=["single", "square4", "line8"])   # N = 1, 4, 8
    a = ap.parse_args()
    if a.device.startswith("cuda"): torch.cuda.set_per_process_memory_fraction(0.6)
    print("uptime:", subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip())
    for B in a.B:
        for cfg in a.configs:
            N, med, best = bench(cfg, B, a.device)
            print(f"{cfg:8s} N={N} B={B}: 中位數 {med / 1e3:9.1f} k module-steps/s，最大 {best / 1e3:9.1f} k"
                  f"（{B / (med / N) * 1e3:.2f} ms / env.step）", flush=True)
    if a.device.startswith("cuda"): print("peak VRAM (GB):", torch.cuda.max_memory_allocated() / 1e9)
    print("uptime:", subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip())
