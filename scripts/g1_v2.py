"""G1 v2：拆開「k 跳資訊不足」和「推力飽和／yaw 權限不足」。（第一版 results/p2/g1.md 保留不動）

相對第一版的改動（白方第 1 輪）：
  1. 最強古典法加入 B2-QP、B2-QP+adapt（k 跳球內的有界最小平方分配，見 quadsim/controllers.py）。
  2. 構型分組（依結構量「直徑」，事先定好）：
       鑑別組 line5、line6、line7、line8、grid10（直徑 ≥ 5 模組，k=1、2 看不到全貌）
       陰性對照組 grid9、plus5（直徑 ≤ 4，預期 k 跳就夠）
  3. 正規化分母 = max(B1, B1-QP)（逐格取平均分數較高者）；分數定義不變。
  4. T2 的目標序列逐 episode 成對（env.py：由 seed、環境槽、episode 編號雜湊決定）。
  5. 診斷（只在鑑別組、T2）：
       yaw0：yaw 目標固定為 0
       twr225：推力上限放大到推重比 2.25（模組用 CAD 質量 0.0381 kg，現行 Tmax=0.149 N 對應推重比只有 1.60，
               2.25 對應 Tmax = 0.210 N；外迴路的 α 限幅也跟著用新的推力上限重算）
       both：兩者同時（額外加的，用來看剩下的是不是純 k 跳資訊）
  6. 最強古典法 = 每格從 {B2+adapt, B2-QP, B2-QP+adapt} × {k=1, k=2} 六個中取平均分數最高者（看評估結果挑的，對古典法有利）。
指標定義同 g1_gate.py。
執行：CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \\
        .venv/bin/python -m scripts.g1_v2 [--n-eps 100] [--tag ""] [--configs ..] [--tasks ..] [--no-diag] [--report-only]"""
import argparse
import json
import math
import time
import numpy as np

from scripts.g1_gate import (RES, ALL_CONFIGS, E_REF, GAP_PP, run_cell, load_gains, fmt)
import quadsim.assembly  # noqa: F401

DISCRIM = ["line5", "line6", "line7", "line8", "grid10"]
CONTROL = ["grid9", "plus5"]
TASKS = ["T2", "POS_STEP", "ATT_STEP"]
DENOM = ["B1", "B1-QP"]
B2_ADAPT = ["B2+adapt k=1", "B2+adapt k=2"]
B2_QP = ["B2-QP k=1", "B2-QP k=2", "B2-QP+adapt k=1", "B2-QP+adapt k=2"]
METHODS = DENOM + B2_ADAPT + B2_QP
CONDS = {"base": dict(), "yaw0": dict(yaw0=True), "twr225": dict(twr=2.25), "both": dict(yaw0=True, twr=2.25)}
COND_DESC = {"base": "基準（G1 設定）", "yaw0": "(i) yaw 目標固定為 0", "twr225": "(ii) 推力上限放大到推重比 2.25",
             "both": "(i)+(ii) 同時"}


def run_all(args, raw_path):
    gains = load_gains()
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}
    meta = dict(gains=gains.to_dict(), n_eps=args.n_eps)
    if raw.get("_meta") != meta: raw = {"_meta": meta}
    configs = args.configs or DISCRIM + CONTROL
    tasks = args.tasks or TASKS
    todo = [("base", c, m, t) for c in configs for t in tasks for m in METHODS]
    if not args.no_diag and "T2" in tasks:
        todo += [(cd, c, m, "T2") for cd in ("yaw0", "twr225", "both") for c in configs if c in DISCRIM for m in METHODS]
    todo = [x for x in todo if "|".join(x) not in raw]
    t_start = time.time()
    for i, (cd, c, m, t) in enumerate(todo):
        t0 = time.time()
        r = raw["|".join((cd, c, m, t))] = run_cell(ALL_CONFIGS[c], m, t, args.n_eps, gains, **CONDS[cd])
        raw_path.write_text(json.dumps(raw))
        eta = (time.time() - t_start) / (i + 1) * (len(todo) - i - 1) / 60
        print(f"[{i + 1}/{len(todo)}] {cd:6s} {c:7s} {m:16s} {t:8s} succ={np.mean(r['success']):.2f} "
              f"score={np.mean(r['score']):.3f} ({time.time() - t0:.0f}s, 剩約 {eta:.0f} 分)", flush=True)
    return raw, gains


# ============================================================================
def get(raw, cd, c, m, t, key="score"): return np.array(raw[f"{cd}|{c}|{m}|{t}"][key], float)


def has(raw, cd, c, t, methods=METHODS): return all(f"{cd}|{c}|{m}|{t}" in raw for m in methods)


def denom_of(raw, cd, c, t):
    """分母 = B1、B1-QP 中平均分數較高者。回傳 (方法名, 每 episode 分數)。"""
    a, b = get(raw, cd, c, "B1", t), get(raw, cd, c, "B1-QP", t)
    return ("B1", a) if a.mean() >= b.mean() else ("B1-QP", b)


def gap(raw, cd, c, m, t, rng=None):
    """落後 pp = 100(1 − mean(分數)/mean(分母))；有 rng 時加成對 bootstrap 95% CI。回傳 (點估計, 下, 上)。"""
    _, d = denom_of(raw, cd, c, t)
    a = get(raw, cd, c, m, t)
    if d.mean() < 0.05: return (math.nan,) * 3
    pt = 100 * (1 - a.mean() / d.mean())
    if rng is None: return pt, math.nan, math.nan
    idx = rng.integers(0, len(a), (1000, len(a)))
    boot = 100 * (1 - a[idx].mean(1) / np.maximum(d[idx].mean(1), 1e-9))
    return pt, *np.percentile(boot, [2.5, 97.5])


def best_of(raw, cd, c, t, pool):
    m = max(pool, key=lambda x: get(raw, cd, c, x, t).mean())
    return m


def cell_rows(raw, groups, rng):
    lines = ["| 構型 | 任務 | 分母(分數/成功率) | " + " | ".join(f"{m} 落後pp" for m in B2_ADAPT + B2_QP) +
             " | 最強古典法 [95% CI] | 其中只含 B2+adapt | 成功率落後 |", "|" + "---|" * (3 + len(B2_ADAPT + B2_QP) + 3)]
    stats = []
    for c in groups:
        for t in TASKS:
            if not has(raw, "base", c, t): continue
            dn, d = denom_of(raw, "base", c, t)
            ds = 100 * get(raw, "base", c, dn, t, "success").mean()
            g = {m: gap(raw, "base", c, m, t)[0] for m in B2_ADAPT + B2_QP}
            bm = best_of(raw, "base", c, t, B2_ADAPT + B2_QP)
            bp, lo, hi = gap(raw, "base", c, bm, t, rng)
            am = best_of(raw, "base", c, t, B2_ADAPT)
            ap = g[am]
            sm = 100 * get(raw, "base", c, bm, t, "success").mean()
            stats.append((c, t, bp, lo, ap))
            lines.append(f"| {c} | {t} | {dn} {d.mean():.3f} / {ds:.0f}% | " + " | ".join(fmt(g[m], 1) for m in B2_ADAPT + B2_QP) +
                         f" | **{fmt(bp, 1)}** ({bm.replace('B2-QP', 'QP').replace('B2+adapt', 'adapt')}) [{fmt(lo, 1)}, {fmt(hi, 1)}] | {fmt(ap, 1)} | {ds - sm:.0f} |")
    return "\n".join(lines), stats


def method_table(raw, groups):
    lines = ["| 構型 | 任務 | " + " | ".join(METHODS) + " |", "|" + "---|" * (2 + len(METHODS))]
    for c in groups:
        for t in TASKS:
            if not has(raw, "base", c, t): continue
            lines.append(f"| {c} | {t} | " + " | ".join(
                f"{get(raw, 'base', c, m, t).mean():.3f} / {100 * get(raw, 'base', c, m, t, 'success').mean():.0f}%" for m in METHODS) + " |")
    return "\n".join(lines)


def diag_table(raw, rng):
    lines = ["| 構型 | 條件 | 分母 (分數 / 成功率) | 最強古典法 (分數 / 成功率) | 落後pp [95% CI] | 只含 B2+adapt 落後pp | QP 家族比 B2+adapt 多補回pp |",
             "|---|---|---|---|---|---|---|"]
    agg = {cd: [] for cd in CONDS}
    for c in DISCRIM:
        for cd in CONDS:
            if not has(raw, cd, c, "T2"): continue
            dn, d = denom_of(raw, cd, c, "T2")
            bm = best_of(raw, cd, c, "T2", B2_ADAPT + B2_QP)
            bp, lo, hi = gap(raw, cd, c, bm, "T2", rng)
            am = best_of(raw, cd, c, "T2", B2_ADAPT)
            ap = gap(raw, cd, c, am, "T2")[0]
            agg[cd].append((bp, ap))
            lines.append(f"| {c if cd == 'base' else ''} | {COND_DESC[cd]} | {dn} {d.mean():.3f} / {100 * get(raw, cd, c, dn, 'T2', 'success').mean():.0f}% | "
                         f"{bm} {get(raw, cd, c, bm, 'T2').mean():.3f} / {100 * get(raw, cd, c, bm, 'T2', 'success').mean():.0f}% | "
                         f"{fmt(bp, 1)} [{fmt(lo, 1)}, {fmt(hi, 1)}] | {fmt(ap, 1)} | {fmt(ap - bp, 1)} |")
    lines.append("")
    lines.append("鑑別組 T2 五個構型的平均（各構型先對自己條件下的分母取落後 pp，再取平均）：")
    lines.append("")
    lines.append("| 條件 | 最強古典法落後pp | 只含 B2+adapt 落後pp |\n|---|---|---|")
    mean = {}
    for cd, v in agg.items():
        if len(v) == len(DISCRIM):
            mean[cd] = (np.nanmean([x[0] for x in v]), np.nanmean([x[1] for x in v]))
            lines.append(f"| {COND_DESC[cd]} | {mean[cd][0]:.1f} | {mean[cd][1]:.1f} |")
    return "\n".join(lines), mean


def decomposition(mean):
    if len(mean) < 4: return "（診斷未跑完，無法拆解。）"
    a_base, q_base = mean["base"][1], mean["base"][0]
    q_yaw, q_twr, q_both = mean["yaw0"][0], mean["twr225"][0], mean["both"][0]
    rows = [("基準：只含 B2+adapt（G1 v1 的最強古典法）", a_base),
            ("　補上飽和處理（B2-QP 家族）後", q_base),
            ("　再把 yaw 目標固定為 0 後", q_yaw),
            ("　再把推力上限放大到推重比 2.25 後（與 yaw0 擇一）", q_twr),
            ("　兩者都做後 = 剩下的 k 跳資訊落後", q_both)]
    t = ["| 階段 | 鑑別組 T2 平均落後pp | 與上一階段差 |", "|---|---|---|"]
    t.append(f"| {rows[0][0]} | {a_base:.1f} | — |")
    t.append(f"| 補上飽和處理（B2-QP 家族，基準環境） | {q_base:.1f} | {q_base - a_base:+.1f} |")
    t.append(f"| 基準 + yaw 固定為 0 | {q_yaw:.1f} | {q_yaw - q_base:+.1f}（相對 QP 基準） |")
    t.append(f"| 基準 + 推力上限放大到推重比 2.25 | {q_twr:.1f} | {q_twr - q_base:+.1f}（相對 QP 基準） |")
    t.append(f"| 兩者同時 | {q_both:.1f} | {q_both - q_base:+.1f}（相對 QP 基準） |")
    return "\n".join(t)


def verdict(raw, stats_d, stats_c):
    ok = [s for s in stats_d if not math.isnan(s[2])]
    n = len(ok); n5 = sum(s[2] >= GAP_PP for s in ok); nci = sum(s[3] > 0 and s[2] >= GAP_PP for s in ok)
    t2 = [s for s in ok if s[1] == "T2"]; t2_5 = sum(s[2] >= GAP_PP for s in t2)
    old = sum(s[4] >= GAP_PP for s in ok)
    mx = max(ok, key=lambda s: s[2])
    ctl = [s for s in stats_c if not math.isnan(s[2])]
    txt = [f"鑑別組 {len(DISCRIM)} 構型 × {len(TASKS)} 任務 = {n} 格。最強古典法（B2+adapt 與 B2-QP（+adapt）中 k=1 或 2 最好者）"
           f"相對分母 max(B1, B1-QP)：",
           f"- 落後 ≥ {GAP_PP:.0f} pp 的有 **{n5} / {n}** 格（其中 95% CI 下界 > 0 的 {nci} 格）；T2 為 {t2_5} / {len(t2)} 格。",
           f"- 最大落後 {mx[2]:.1f} pp（{mx[0]} {mx[1]}），中位數 {np.median([s[2] for s in ok]):.1f} pp。",
           f"- 對照：只用 B2+adapt（沒有 QP）時 ≥ {GAP_PP:.0f} pp 的格數是 {old} / {n}。",
           f"- 陰性對照組（grid9、plus5）最大落後 {max(s[2] for s in ctl):.1f} pp。" if ctl else ""]
    return n5, n, "\n".join(x for x in txt if x)


def write_report(raw, gains, args, path):
    rng = np.random.default_rng(0)
    tab_d, stats_d = cell_rows(raw, DISCRIM, rng)
    tab_c, stats_c = cell_rows(raw, CONTROL, rng)
    dtab, mean = diag_table(raw, rng) if has(raw, "base", DISCRIM[0], "T2") and has(raw, "yaw0", DISCRIM[0], "T2") else ("（診斷未跑）", {})
    n5, n, vtxt = verdict(raw, stats_d, stats_c)
    g = gains.to_dict()
    notes = RES / f"g1_v2_notes{args.tag}.md"
    md = [f"# G1 v2：拆開 k 跳資訊與推力飽和（{time.strftime('%Y-%m-%d')}）", "",
          f"由 `scripts/g1_v2.py` 產生（原始資料 `g1_v2_raw{args.tag}.json`）。每格 {raw['_meta']['n_eps']} 個 episode，固定 seed，"
          "同一構型與任務下所有方法的初始條件相同；T2 的目標序列逐 episode 成對（由 seed、環境槽、episode 編號決定）。"
          "標稱環境（沒有 payload、對接偏差、推力誤差）。第一版 `g1.md` 保留不動。", "",
          f"外迴路增益同 g1.md（kp={g['kp']:.2f}、kd={g['kd']:.2f}、kR={g['kR']:.1f}、kw={g['kw']:.1f}、yaw 倍率 {g['yaw_ratio']:.2f}）。", "",
          "## 構型分組（依結構量「直徑」事先決定）", "",
          f"- **鑑別組**：{', '.join(DISCRIM)}（直徑 ≥ 5 模組，k=1、2 球看不到全貌；H1 只在這組判定）",
          f"- **陰性對照組**：{', '.join(CONTROL)}（直徑 ≤ 4，預期 k 跳就夠；照樣報告）",
          "- 15 格（5 構型 × 3 任務）以「任一格 ≥ 5pp」判定沒有做多重比較修正，prereg 要說明；這裡同時報 CI 下界。", "",
          "## 判定", "", vtxt, "",
          f"定義：分母 = 每格 max(B1, B1-QP)（平均分數較高者）；落後 pp = 100 × (1 − 方法平均分數 / 分母平均分數)，負值 = 比分母好；"
          "最強古典法 = 每格 6 個方法（B2+adapt、B2-QP、B2-QP+adapt × k=1、2）中平均分數最高者（看評估結果挑，對古典法有利）；"
          "CI 為對 episode 的成對 bootstrap（1000 次）。", "",
          "## 診斷（鑑別組、T2）", "", dtab, "",
          "### 落後的來源拆解", "", decomposition(mean), "",
          notes.read_text() if notes.exists() else "（尚無文字解讀）", "",
          "## 主表：鑑別組（落後 pp，基準環境）", "", tab_d, "",
          "## 主表：陰性對照組", "", tab_c, "",
          "## 各方法的 平均分數 / 成功率", "", "### 鑑別組", "", method_table(raw, DISCRIM), "",
          "### 陰性對照組", "", method_table(raw, CONTROL), ""]
    path.write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eps", type=int, default=100)
    ap.add_argument("--tag", default="")
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--no-diag", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    raw_path = RES / f"g1_v2_raw{args.tag}.json"
    if args.report_only: raw, gains = json.loads(raw_path.read_text()), load_gains()
    else: raw, gains = run_all(args, raw_path)
    write_report(raw, gains, args, RES / f"g1_v2{args.tag}.md")
    print("已寫入", RES / f"g1_v2{args.tag}.md")


if __name__ == "__main__":
    main()
