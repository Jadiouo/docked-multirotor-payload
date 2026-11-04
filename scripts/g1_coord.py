"""G1 coord：協調飽和的古典法 B2-QP-coord 的公平性關卡（RL 探針之前）。

流程：
  1) tune：B1-QP-yawlim 在 TRAIN 構型（single、line2、line3、square4、S4）的 T2 上做 20 組隨機搜尋
     （yaw 速率係數 rate_c、yaw_ratio），結果存 results/p2/g1_coord_tune.json；B2-QP-coord-yawlim 沿用同一組參數。
  2) eval：和 g1_v2 同一套設定（每格 n_eps、同 seed），方法與 g1_v2_raw.json 的結果直接比較（不重跑）。
  3) report：results/p2/g1_coord.md 與 g1_coord_gap_vs_length.png。
執行：CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
        .venv/bin/python -m scripts.g1_coord {tune|eval|report} [--n-eps 100] [--stage t2|rest]"""
import argparse, dataclasses, json, math, time
import numpy as np
from scripts.g1_gate import RES, ALL_CONFIGS, run_cell, load_gains, fmt
import scripts.g1_gate as gg
from scripts.tune_classical import TRAIN_CONFIGS
from scripts import g1_v2 as v2

TUNE = RES / "g1_coord_tune.json"
RAW = RES / "g1_coord_raw.json"
V2RAW = RES / "g1_v2_raw.json"
DISCRIM, CONTROL, TASKS = v2.DISCRIM, v2.CONTROL, v2.TASKS
VARS = ["v1", "v2", "v3"]
COORD = [f"B2-QP-coord-{v}{a} k={k}" for v in VARS for a in ("", "+adapt") for k in (1, 2)]
YAWLIM = ["B1-QP-yawlim"] + [f"B2-QP-coord-{v}+adapt-yawlim k=2" for v in VARS]
NEW = COORD + YAWLIM


def tune(n_eps=32, n_trials=20):
    base = load_gains()
    rng = np.random.default_rng(123)
    trials = [dict(rate_c=1e3, yaw_ratio=base.yaw_ratio, note="無限速（基準）")]
    trials += [dict(rate_c=float(np.exp(rng.uniform(np.log(0.3), np.log(10)))), yaw_ratio=float(rng.uniform(0.05, 0.5)), note="") for _ in range(n_trials)]
    out = []
    for i, t in enumerate(trials):
        g = dataclasses.replace(base, yaw_ratio=t["yaw_ratio"])
        sc = [np.mean(run_cell(c, "B1-QP-yawlim", "T2", n_eps, g, ckw=dict(rate_c=t["rate_c"]))["score"]) for c in TRAIN_CONFIGS.values()]
        out.append(dict(t, score=float(np.mean(sc)), per_config=[float(x) for x in sc]))
        print(f"[tune {i}] rate_c={t['rate_c']:.2f} yaw_ratio={t['yaw_ratio']:.3f} TRAIN-T2={np.mean(sc):.4f}", flush=True)
    best = max(out[1:], key=lambda x: x["score"])           # 20 組搜尋的最佳（基準另列）
    TUNE.write_text(json.dumps(dict(trials=out, best=best, n_eps=n_eps, configs=list(TRAIN_CONFIGS)), indent=1))
    print("best", best)


def method_args(m, gains):
    if "yawlim" not in m: return gains, {}
    b = json.loads(TUNE.read_text())["best"]
    return dataclasses.replace(gains, yaw_ratio=b["yaw_ratio"]), dict(rate_c=b["rate_c"])


def evaluate(n_eps, stage):
    gains = load_gains()
    raw = json.loads(RAW.read_text()) if RAW.exists() else {}
    meta = dict(gains=gains.to_dict(), n_eps=n_eps, tune=json.loads(TUNE.read_text())["best"])
    if raw.get("_meta") != meta: raw = {"_meta": meta}
    if stage == "t2": todo = [(c, m, "T2") for c in DISCRIM for m in NEW]
    else: todo = [(c, m, t) for c in DISCRIM + CONTROL for t in TASKS for m in NEW if not (c in DISCRIM and t == "T2")]
    todo = [x for x in todo if "|".join(x) not in raw]
    t0 = time.time()
    for i, (c, m, t) in enumerate(todo):
        g, ck = method_args(m, gains)
        t1 = time.time()
        r = raw["|".join((c, m, t))] = run_cell(ALL_CONFIGS[c], m, t, n_eps, g, ckw=ck)
        RAW.write_text(json.dumps(raw))
        print(f"[{i + 1}/{len(todo)}] {c:7s} {m:36s} {t:8s} succ={np.mean(r['success']):.2f} score={np.mean(r['score']):.3f} "
              f"({time.time() - t1:.0f}s, 剩約 {(time.time() - t0) / (i + 1) * (len(todo) - i - 1) / 60:.0f} 分)", flush=True)


# ------------------------------------------------------------------ 報告
def sc(raw, v2raw, c, m, t):
    k = f"{c}|{m}|{t}"
    if k in raw: return np.array(raw[k]["score"], float)
    k = f"base|{c}|{m}|{t}"
    return np.array(v2raw[k]["score"], float) if k in v2raw else None


def sucs(raw, v2raw, c, m, t):
    k = f"{c}|{m}|{t}"
    d = raw[k] if k in raw else v2raw.get(f"base|{c}|{m}|{t}")
    return None if d is None else np.mean(d["success"])


def gap_pp(a, d):
    return math.nan if a is None or d.mean() < 0.05 else 100 * (1 - a.mean() / d.mean())


def gap_ci(a, d, rng):
    idx = rng.integers(0, len(a), (1000, len(a)))
    b = 100 * (1 - a[idx].mean(1) / np.maximum(d[idx].mean(1), 1e-9))
    return np.percentile(b, [2.5, 97.5])


def report(n_eps):
    raw, v2raw = json.loads(RAW.read_text()), json.loads(V2RAW.read_text())
    tune = json.loads(TUNE.read_text())
    rng = np.random.default_rng(0)
    G = lambda c, m, t: sc(raw, v2raw, c, m, t)
    def denom(c, t):
        a, b = G(c, "B1", t), G(c, "B1-QP", t)
        return ("B1", a) if a.mean() >= b.mean() else ("B1-QP", b)
    V2_POOL = v2.B2_ADAPT + v2.B2_QP
    def best_v2(c, t): return max(V2_POOL, key=lambda m: G(c, m, t).mean())
    def best_of(c, t, pool):
        pool = [m for m in pool if G(c, m, t) is not None]
        return max(pool, key=lambda m: G(c, m, t).mean())
    have = lambda c, t: G(c, NEW[0], t) is not None and all(G(c, m, t) is not None for m in NEW)
    L = ["# G1 coord：協調飽和的古典法（B2-QP-coord）能補回多少缺口", "",
         f"由 `scripts/g1_coord.py` 產生（原始資料 `g1_coord_raw.json`；g1_v2 的方法直接讀 `g1_v2_raw.json`，沒有重跑）。每格 {n_eps} 個 episode、同 seed、成對比較，"
         "標稱環境；分母 = 每格 max(B1, B1-QP)；落後 pp = 100(1 − 方法平均分數/分母平均分數)。", "",
         "## 事先規定（在看到結果之前定下來）", "",
         "協調機制：各模組照 B2-QP 解出自己 k 跳球內的分配，回報 4 個純量（Fz、roll、pitch、yaw 各自「達成的 wrench / 要求的 wrench」，夾在 [0,1]）；"
         "leader 對所有模組取 min，得到 4 個純量縮放比，廣播給所有模組；各模組把要求乘上縮放比後重解。"
         "每個模組收到的資訊 = B2 原有的（f_acc、α_cmd、自己 k 跳球的形狀）+ 每回合 4 個純量；leader 收到每個模組 4 個純量。**不廣播形狀。**沒有飽和時縮放比 = 1，輸出等於 B2-QP（有測試）。", "",
         "| 變體 | 縮放方式 | 回合數 | 廣播純量 |", "|---|---|---|---|",
         "| v1 | 每個軸（Fz、roll、pitch、yaw）各自一個全域縮放比 | 1 | 4 |",
         "| v2 | 優先序：只縮 yaw；roll、pitch、Fz 任何一軸有模組達不到，就繼續縮 yaw（縮放比 = 所有模組、所有軸達成比的最小值），保 roll/pitch/Fz | 2 | 4 / 回合 |",
         "| v3 | 保方向：roll、pitch、yaw 三個力矩共用同一個縮放比（維持力矩向量的方向，Fz 不縮），縮放比 = 所有模組的三軸最小達成比，連續縮放 | 2 | 4 / 回合（三個力矩分量相同） |", "",
         "v3 是我自己設計的方案。「最好的一種」是看評估結果挑的，對古典法有利，沒有修正。", "",
         "另一個古典對手 **B1-QP-yawlim**：B1-QP + yaw 參考速率限制（速率上限 = rate_c·√(α_yaw 權限)，權限由標稱形狀算，是依構型 yaw 權限的增益排程）"
         f"，並同時調 yaw_ratio。在 TRAIN 構型（{', '.join(tune['configs'])}）的 T2 上做 20 組隨機搜尋（每組 {tune['n_eps']} episodes），"
         f"選到 rate_c = {tune['best']['rate_c']:.2f}、yaw_ratio = {tune['best']['yaw_ratio']:.3f}（TRAIN T2 分數 {tune['best']['score']:.4f}；"
         f"無限速基準 {tune['trials'][0]['score']:.4f}）。B2-QP-coord-*-yawlim 沿用同一組參數、不再另調（局部形狀 + 同樣的 yaw 限速）。", ""]
    # ---- 主表
    def table(groups, tasks):
        hdr = ["B2-QP+adapt k=2", "g1_v2 最強古典法"] + [f"coord-{v}+adapt k=2" for v in VARS] + ["最強 coord (k,adapt,變體)", "B1-QP-yawlim"] + [f"coord-{v}-yawlim" for v in VARS]
        t = ["| 構型 | 任務 | 分母(分數/成功率) | " + " | ".join(h + " 落後pp" for h in hdr) + " |", "|" + "---|" * (3 + len(hdr))]
        rows = []
        for c in groups:
            for tk in tasks:
                if not have(c, tk): continue
                dn, d = denom(c, tk)
                bv = best_v2(c, tk); bc = best_of(c, tk, COORD)
                cells = [gap_pp(G(c, "B2-QP+adapt k=2", tk), d), gap_pp(G(c, bv, tk), d)] + [gap_pp(G(c, f"B2-QP-coord-{v}+adapt k=2", tk), d) for v in VARS]
                lo, hi = gap_ci(G(c, bc, tk), d, rng)
                cells += [gap_pp(G(c, bc, tk), d), gap_pp(G(c, "B1-QP-yawlim", tk), d)] + [gap_pp(G(c, f"B2-QP-coord-{v}+adapt-yawlim k=2", tk), d) for v in VARS]
                rows.append((c, tk, bv, bc, cells))
                t.append(f"| {c} | {tk} | {dn} {d.mean():.3f} / {100 * np.mean(raw_succ(c, dn, tk)):.0f}% | " +
                         " | ".join(fmt(x, 1) for x in cells[:2]) + " | " + " | ".join(fmt(x, 1) for x in cells[2:5]) +
                         f" | {fmt(cells[5], 1)} [{fmt(lo, 1)}, {fmt(hi, 1)}] ({bc.replace('B2-QP-coord-', '')}) | " + " | ".join(fmt(x, 1) for x in cells[6:]) + " |")
        return "\n".join(t), rows
    def raw_succ(c, m, t):
        d = v2raw[f"base|{c}|{m}|{t}"]; return d["success"]
    L += ["## 主表（落後 pp，越小越好；負值 = 比分母好）", "",
          "欄位：「g1_v2 最強古典法」= B2+adapt、B2-QP(+adapt) × k=1,2 六個中該格最好者（沿用 g1_v2 的定義）；「最強 coord」= 12 個 coord 方法（3 變體 × k=1,2 × 有無 adapt）中該格最好者。", ""]
    td, rows_d = table(DISCRIM, TASKS); tc, rows_c = table(CONTROL, TASKS)
    L += ["### 鑑別組", "", td, "", "### 陰性對照組", "", tc, ""]
    # ---- 補回
    L += ["## B2-QP-coord 補回了 g1_v2 缺口的幾 %", "",
          "補回 % = (g1_v2 最強古典法落後 − 方法落後) / g1_v2 最強古典法落後 × 100，只在 g1_v2 落後 ≥ 5 pp 的格計算（避免除以接近 0 的數）。"
          "「合併」= 這些格的落後 pp 總和算比例。", ""]
    big = [r for r in rows_d if r[4][1] >= 5.0]
    def rec(idx_name, get):
        if not big: return "（沒有 ≥ 5 pp 的格）"
        return None
    meths = [("coord v1 (k=2,adapt)", 2), ("coord v2 (k=2,adapt)", 3), ("coord v3 (k=2,adapt)", 4), ("最強 coord（逐格挑）", 5),
             ("B1-QP-yawlim", 6), ("coord-v1-yawlim", 7), ("coord-v2-yawlim", 8), ("coord-v3-yawlim", 9)]
    t = ["| 方法 | " + " | ".join(f"{r[0]} {r[1]}" for r in big) + " | 合併補回 % | 鑑別組 T2 平均落後 pp |", "|" + "---|" * (3 + len(big))]
    for name, ix in meths:
        base_tot = sum(r[4][1] for r in big); tot = sum(r[4][ix] for r in big)
        t2 = [r[4][ix] for r in rows_d if r[1] == "T2"]
        t.append(f"| {name} | " + " | ".join(f"{100 * (r[4][1] - r[4][ix]) / r[4][1]:.0f}%" for r in big) + f" | **{100 * (base_tot - tot) / base_tot:.0f}%** | {np.mean(t2):.1f} |")
    t2b = [r[4][1] for r in rows_d if r[1] == "T2"]
    L += ["\n".join(t), "", f"（對照：g1_v2 最強古典法在鑑別組 T2 平均落後 {np.mean(t2b):.1f} pp；B2-QP+adapt k=2 為 {np.mean([r[4][0] for r in rows_d if r[1] == 'T2']):.1f} pp。）", ""]
    # ---- 剩下的缺口
    L += ["## 剩下的缺口（每格取所有古典方法中最好者，含 yawlim）", "",
          "| 構型 | 任務 | 最好的新方法 | 落後 pp [95% CI] | 對照：相對 max(分母, B1-QP-yawlim) 的落後 pp |", "|---|---|---|---|---|"]
    rem = []
    for c in DISCRIM:
        for tk in TASKS:
            if not have(c, tk): continue
            dn, d = denom(c, tk)
            bn = best_of(c, tk, NEW); lo, hi = gap_ci(G(c, bn, tk), d, rng)
            yl = G(c, "B1-QP-yawlim", tk); d2 = yl if yl.mean() > d.mean() else d
            rem.append((c, tk, gap_pp(G(c, bn, tk), d), gap_pp(G(c, bn, tk), d2)))
            L.append(f"| {c} | {tk} | {bn.replace('B2-QP-coord-', 'coord-')} | {fmt(rem[-1][2], 1)} [{fmt(lo, 1)}, {fmt(hi, 1)}] | {fmt(rem[-1][3], 1)} |")
    L += [""]
    # ---- 圖
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        lines = ["line5", "line6", "line7", "line8"]
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
        series = [("g1_v2 最強古典法", lambda c, tk: best_v2(c, tk)), ("B2-QP+adapt k=2", lambda c, tk: "B2-QP+adapt k=2")] + \
                 [(f"coord-{v}+adapt k=2", (lambda v: lambda c, tk: f"B2-QP-coord-{v}+adapt k=2")(v)) for v in VARS] + \
                 [("B1-QP-yawlim", lambda c, tk: "B1-QP-yawlim")] + [(f"coord-{v}-yawlim", (lambda v: lambda c, tk: f"B2-QP-coord-{v}+adapt-yawlim k=2")(v)) for v in VARS]
        for ax, tk in zip(axs, TASKS):
            cs = [c for c in lines if have(c, tk)]
            for name, f in series:
                ax.plot([5 + lines.index(c) for c in cs], [gap_pp(G(c, f(c, tk), tk), denom(c, tk)[1]) for c in cs], marker="o", label=name, ls="--" if "yawlim" in name else "-")
            ax.set_title(f"{tk}"); ax.set_xlabel("line 模組數"); ax.set_ylabel("落後 pp（相對 max(B1,B1-QP)）"); ax.axhline(0, c="k", lw=.5); ax.set_xticks([5, 6, 7, 8])
        axs[0].legend(fontsize=7)
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"]
        fig.tight_layout(); fig.savefig(RES / "g1_coord_gap_vs_length.png", dpi=130)
        L += ["## 缺口 vs 構型長度", "", "![gap vs length](g1_coord_gap_vs_length.png)", "", "（line5 → line8；實線 = 無 yaw 限速；虛線 = 含 yaw 限速。）", ""]
    except Exception as e:
        L += [f"（圖未產生：{e}）", ""]
    # ---- 其餘各方法分數
    L += ["## 各方法 平均分數 / 成功率（新方法；g1_v2 方法見 g1_v2.md）", "",
          "| 構型 | 任務 | B1-QP | " + " | ".join(m.replace("B2-QP-coord-", "c-") for m in NEW) + " |", "|" + "---|" * (3 + len(NEW))]
    for c in DISCRIM + CONTROL:
        for tk in TASKS:
            if have(c, tk):
                L.append(f"| {c} | {tk} | {G(c, 'B1-QP', tk).mean():.3f} | " + " | ".join(f"{G(c, m, tk).mean():.3f} / {100 * sucs(raw, v2raw, c, m, tk):.0f}%" for m in NEW) + " |")
    L += ["", "## 事實陳述", "", FACTS(rows_d) if callable(FACTS) else FACTS]
    (RES / "g1_coord.md").write_text("\n".join(L))
    print("已寫入", RES / "g1_coord.md")


FACTS = "（待填）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "eval", "report"])
    ap.add_argument("--n-eps", type=int, default=100)
    ap.add_argument("--stage", default="t2", choices=["t2", "rest"])
    a = ap.parse_args()
    if a.cmd == "tune": tune()
    elif a.cmd == "eval": evaluate(a.n_eps, a.stage)
    else:
        facts = RES / "g1_coord_facts.md"
        global FACTS
        if facts.exists(): FACTS = facts.read_text()
        report(a.n_eps)


if __name__ == "__main__":
    main()
