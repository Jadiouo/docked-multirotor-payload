"""比較手寫 PPO 與 CleanRL 在 Hopper-v5 的結果（P1 驗收）。

讀 results/p1/hopper_{cleanrl,ours}_seed{S}.csv（欄位 global_step, episodic_return），
算每個 seed「最後 10 萬步」的平均 episodic return，印表格，並畫學習曲線到
results/p1/hopper_compare.png。
用法：python scripts/compare_hopper.py --seeds 1 2 3
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {"cleanrl": "CleanRL", "ours": "ours (ppo.py)"}
COLORS = {"cleanrl": "tab:blue", "ours": "tab:orange"}


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return (np.array([float(r["global_step"]) for r in rows]),
            np.array([float(r["episodic_return"]) for r in rows]))


def smooth_curve(steps, rets, grid, window):
    """在 grid 的每個點上，取前 window 步內結束的 episode 平均回報（沒有就沿用前一個值）。"""
    out, last = [], np.nan
    for g in grid:
        m = (steps <= g) & (steps > g - window)
        if m.any():
            last = rets[m].mean()
        out.append(last)
    return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--dir", default="results/p1")
    p.add_argument("--final-window", type=int, default=100_000)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or os.path.join(args.dir, "hopper_compare.png")

    data = {k: {} for k in LABELS}
    for k in LABELS:
        for s in args.seeds:
            path = os.path.join(args.dir, f"hopper_{k}_seed{s}.csv")
            if os.path.exists(path):
                data[k][s] = load(path)
            else:
                print(f"[warn] 缺檔 {path}")

    finals = {k: {} for k in LABELS}
    print(f"{'impl':<14}{'seed':>5}{'last_step':>11}{'episodes':>9}{'final_mean_return':>19}")
    for k in LABELS:
        for s, (steps, rets) in sorted(data[k].items()):
            last = steps.max()
            m = steps > last - args.final_window
            finals[k][s] = rets[m].mean()
            print(f"{LABELS[k]:<14}{s:>5}{int(last):>11}{int(m.sum()):>9}{finals[k][s]:>19.1f}")
    for k in LABELS:
        v = np.array(list(finals[k].values()))
        if len(v):
            print(f"{LABELS[k]:<14} mean {v.mean():.1f}  min {v.min():.1f}  max {v.max():.1f}")
    if finals["cleanrl"] and finals["ours"]:
        lo, hi = min(finals["cleanrl"].values()), max(finals["cleanrl"].values())
        inside = {s: lo <= x <= hi for s, x in finals["ours"].items()}
        print(f"CleanRL 範圍 [{lo:.1f}, {hi:.1f}]；ours 各 seed 是否落在範圍內：{inside}；"
              f"P1 {'通過' if all(inside.values()) and len(inside) >= 3 else '未通過/未滿 3 seed'}")

    all_last = [st.max() for k in data for st, _ in data[k].values()]
    if not all_last:
        print("沒有任何資料，不畫圖")
        return
    grid = np.linspace(0, max(all_last), 200)
    window = max(20_000, args.final_window // 5)
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in LABELS:
        curves = []
        for s, (steps, rets) in sorted(data[k].items()):
            c = smooth_curve(steps, rets, grid, window)
            curves.append(c)
            ax.plot(grid, c, color=COLORS[k], alpha=0.35, lw=1)
        if curves:
            ax.plot(grid, np.nanmean(curves, axis=0), color=COLORS[k], lw=2.2,
                    label=f"{LABELS[k]} (mean of {len(curves)} seeds)")
    ax.set_xlabel("global step")
    ax.set_ylabel(f"episodic return (mean over last {window // 1000}k steps)")
    ax.set_title("Hopper-v5: ours vs CleanRL (thin = per seed)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    main()
