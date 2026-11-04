"""只在訓練構型上調古典外迴路的增益（調 B1，B2、B5 共用同一組）。

- 訓練構型：single、line2、line3、square4，加上 S4、L4。env 吃任意格點 cells，所以 S4、L4 直接支援
  （quadsim/assembly.py 的 CONFIGS 沒有它們，在這裡定義；兩個都是樹狀）。留出構型（L3、T4、line4 …）一個都不碰。
- 預算 40 組：第 0 組是 Gains() 的預設值，其餘 39 組在對數尺度上均勻隨機抽（固定 seed）。
- 指標：T2 的「每 episode 總 reward」，對 episode 與構型取平均。每個環境只算第一個 episode
  （env 會自動 reset，終止之後的 reward 不計）。調參用的環境 seed 與 G1 評估的 seed 不同。
執行（專案根目錄）：
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 \
    .venv/bin/python -m scripts.tune_classical [--n-envs 32] [--budget 40]"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch

from quadsim.assembly import CONFIGS, TRAIN
from quadsim.env import ModularEnv
from quadsim.controllers import B1, Gains, goal_of

torch.set_num_threads(1)    # 機器忙的時候，多執行緒做這種小矩陣會互相等待而慢十倍以上；單執行緒穩定

OUT = Path(__file__).resolve().parent.parent / "results" / "p2" / "classical_gains.json"
TUNE_SEED = 1000            # 調參的環境 seed；G1 評估用 0..，兩者不重疊

TRAIN_CONFIGS = {n: CONFIGS[n] for n in TRAIN}
TRAIN_CONFIGS["S4"] = [(0, 0), (1, 0), (1, 1), (2, 1)]
TRAIN_CONFIGS["L4"] = [(0, 0), (1, 0), (2, 0), (0, 1)]

# 搜尋範圍（對數均勻）。上限由馬達一階延遲 τ = 0.07 s 決定：姿態迴路自然頻率 √kR 要明顯低於 1/τ ≈ 14 rad/s，
# 阻尼項 kw 太大（> 1/τ 量級）會和馬達延遲、100 Hz 取樣互相作用而振盪，碰到飽和就變成極限環。
# （第一次搜尋用 kR ≤ 300、kw ≤ 40：小構型看起來最好，但 line8 的 roll 軸飽和後來回振盪而墜毀，所以收斂範圍。）
RANGES = dict(kp=(2.0, 20.0), kd=(2.0, 10.0), kR=(10.0, 120.0), kw=(3.0, 20.0), yaw_ratio=(0.05, 1.0))


def episode_return(env, ctrl):
    """每個環境第一個 episode 的總 reward，回傳 [B]。"""
    st = env.state()
    total = torch.zeros(env.B)
    alive = torch.ones(env.B, dtype=torch.bool)
    for _ in range(env.cfg.episode_steps):
        _, r, te, tr, info = env.step(ctrl.act(st, goal_of(st), env))
        st = info["state"]
        total += r * alive
        alive &= ~(te | tr)
    return total


def score(gains, n_envs):
    """一組增益在所有訓練構型上的平均 T2 reward，以及各構型的值。"""
    per = {}
    for name, cells in TRAIN_CONFIGS.items():
        env = ModularEnv(cells, n_envs, task="T2", seed=TUNE_SEED)
        env.reset()
        per[name] = episode_return(env, B1(gains)).mean().item()
    return float(np.mean(list(per.values()))), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--budget", type=int, default=40)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    trials = [Gains()]                                                   # 第 0 組：預設值
    for _ in range(args.budget - 1):
        trials.append(Gains(**{k: float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) for k, (lo, hi) in RANGES.items()}))

    results = []
    for i, g in enumerate(trials):
        t0 = time.time()
        s, per = score(g, args.n_envs)
        results.append(dict(gains=g.to_dict(), score=s, per_config=per))
        print(f"[{i:2d}/{len(trials)}] score={s:8.2f}  kp={g.kp:5.1f} kd={g.kd:5.1f} kR={g.kR:6.1f} kw={g.kw:5.1f} yaw={g.yaw_ratio:4.2f} ({time.time() - t0:.0f}s)", flush=True)

    best = max(range(len(results)), key=lambda i: results[i]["score"])
    print(f"最佳：第 {best} 組，score={results[best]['score']:.2f}（預設 {results[0]['score']:.2f}）")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        gains=results[best]["gains"], score=results[best]["score"], best_index=best, default_score=results[0]["score"],
        configs=list(TRAIN_CONFIGS), n_envs=args.n_envs, seed=TUNE_SEED, budget=args.budget,
        ranges=RANGES, trials=results), indent=1, ensure_ascii=False))
    print("已寫入", OUT)


if __name__ == "__main__":
    main()
