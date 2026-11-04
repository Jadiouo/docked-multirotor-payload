"""Hopper 對照的 bootstrap 區間（論文附錄 Hopper Check 的數字出處）。

每個 seed 取「最後 10 萬步」結束的 episode 平均回報（同 compare_hopper.py），
相對差 = (mean(ours) − mean(CleanRL)) / mean(CleanRL)。區間：兩組各自獨立重抽 seed，100k 次，
分母固定為原樣本的 CleanRL 平均（與驗收規則「平均差的 CI 落在 CleanRL 平均的 ±15% 內」相同），取 2.5／97.5 百分位。
用法：python scripts/hopper_ci.py  →  results/p1/hopper_ci.json
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from compare_hopper import load

B, LAST = 100_000, 100_000


def final(kind, s):
    st, r = load(f"results/p1/hopper_{kind}_seed{s}.csv")
    return float(r[st > st.max() - LAST].mean())


def ci(n, rng):
    o = np.array([final("ours", s) for s in range(1, n + 1)])
    c = np.array([final("cleanrl", s) for s in range(1, n + 1)])
    bo = o[rng.integers(0, n, (B, n))].mean(1)
    bc = c[rng.integers(0, n, (B, n))].mean(1)
    rel = 100 * (bo - bc) / c.mean()
    return dict(n_seeds=n, ours=o.mean(), cleanrl=c.mean(), rel_pct=100 * (o.mean() / c.mean() - 1),
                ci95_pct=[float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))],
                ours_min=o.min(), cleanrl_min=c.min())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    out = {f"seeds_{n}": ci(n, rng) for n in (8, 12)}
    out["_meta"] = dict(B=B, last_steps=LAST, rng="numpy default_rng(0)")
    json.dump(out, open("results/p1/hopper_ci.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
