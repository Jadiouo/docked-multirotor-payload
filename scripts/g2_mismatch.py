"""G2：古典控制在「物理誤差未知」時退化多少？（RL 的利基是不是「免辨識的魯棒性」）

背景：G1 v2 發現，標稱物理下補上飽和處理後 k 跳形狀資訊不是瓶頸。要判斷 RL 的利基還在不在，先量最強古典法
在 payload、推力係數、對接偏差下掉多少。所有方法拿到的都是「標稱」形狀與質量（env.nominal、env.p），
env 的真實幾何（payload、對接偏差、各馬達推力係數）只影響動力學。

構型：TRAIN line3、square4；鑑別組 line6、line8、grid10；對照組 grid9。任務：POS_STEP、T2。
方法（label → 工廠名與參數）：
  B1、B1-QP（不適應）、B1-QP+adapt（逐軸 α 增益 RLS + 總推力增益 RLS）、B2-QP+adapt k=2（同上 + 總推力增益）、
  B3 INDI、B3+adapt（INDI + G 對角縮放 RLS）、B4（B1-QP+adapt + 力矩偏置 b̂ ≈ CoM 偏移）、B5。
  所有 *adapt 方法與 B4 都加了「總推力增益 κ̂」（FzAdapter，用加速度計），否則質量誤差讓高度永遠下沉，對古典法不公平。
  B2-QP+adapt 在 G1 v2 沒有 κ̂，這裡加上（fz_adapt=True），所以數字不能和 g1_v2.md 逐格比。
誤差（每次只開一種，加一組全開）：
  payload 15%、30%（質點，掛在 x 最大的模組中心，佔總質量比例）；推力係數 ±10%、±20%（每顆馬達獨立 U(1−ε,1+ε)）；
  對接偏差 位置 ±5 mm、傾斜 ±2°（每個模組獨立）；全開 = payload 30% + 推力 ±10% + 對接偏差。
  對接偏差與推力係數每個 episode 重抽（同 seed 下所有方法的第一個 episode 幾何相同，成對）；payload 固定。
指標：分數、成功率、接頭負載比的 p95（每個 episode 取最大值，再對 episode 取 95 百分位）。定義同 g1_gate.py / g1_v2.py。
  退化 pp = 100(1 − 該方法在誤差下的平均分數 / 該方法在標稱下的平均分數)（對一個範圍內的格子先各自平均再相除）。
  落後 pp = 每格 100(1 − 方法分數 / 同一格同一條件下最佳方法的分數)，再對範圍內的格子取平均。
執行：CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \\
        .venv/bin/python -m scripts.g2_mismatch [--n-eps 100] [--tag ""] [--configs ..] [--conds ..] [--tasks ..] [--methods ..] [--report-only]"""
import argparse
import json
import math
import time
import numpy as np

from scripts.g1_gate import RES, ALL_CONFIGS, run_cell, load_gains, fmt
from quadsim.env import Perturb
import quadsim.assembly  # noqa: F401

TRAIN_G = ["line3", "square4"]
DISC = ["line6", "line8", "grid10"]
CTRL = ["grid9"]
CONFIGS = TRAIN_G + DISC + CTRL
TASKS = ["POS_STEP", "T2"]
SCOPES = {"全部 6 構型": CONFIGS, "TRAIN（line3、square4）": TRAIN_G, "鑑別組（line6、line8、grid10）": DISC, "對照組（grid9）": CTRL}

# label → (工廠名, 參數)
METHODS = {
    "B1": ("B1", {}),
    "B1-QP": ("B1-QP", {}),
    "B1-QP+adapt": ("B1-QP+adapt", {}),
    "B2-QP+adapt k=2": ("B2-QP+adapt k=2", dict(fz_adapt=True, g_range=(0.5, 2.0))),
    "B3 INDI": ("B3 INDI", {}),
    "B3+adapt": ("B3+adapt", {}),
    "B4": ("B4", {}),
    "B5": ("B5", {}),
}
CONDS = {   # 條件名 → Perturb 參數（payload_host 另依構型填）
    "nominal": dict(),
    "payload10": dict(payload_frac=0.10),     # 補充條件（不在要求清單內）：30% 在 TWR 1.6 下物理上不可行，見「物理可行性」
    "payload15": dict(payload_frac=0.15), "payload30": dict(payload_frac=0.30),
    "thrust10": dict(thrust_eps=0.10), "thrust20": dict(thrust_eps=0.20),
    "dock": dict(dock_delta=0.005, dock_tilt=math.radians(2)),
    "all": dict(payload_frac=0.30, thrust_eps=0.10, dock_delta=0.005, dock_tilt=math.radians(2)),
    "all15": dict(payload_frac=0.15, thrust_eps=0.10, dock_delta=0.005, dock_tilt=math.radians(2)),   # 補充：全開但 payload 15%
}
COND_DESC = {"nominal": "標稱", "payload10": "payload 10%（補充）", "all15": "全開 payload 15%（補充）", "payload15": "payload 15%", "payload30": "payload 30%", "thrust10": "推力係數 ±10%",
             "thrust20": "推力係數 ±20%", "dock": "對接偏差 ±5mm、±2°", "all": "全開（payload 30% + 推力 ±10% + 偏差）"}


def perturb_of(cond, cells):
    kw = dict(CONDS[cond])
    if kw.get("payload_frac", 0) > 0:     # 掛在 x 最大端的模組（有多個時取第一個）
        kw["payload_host"] = int(np.argmax([c[0] for c in cells]))
    return Perturb(**kw)


def key(cond, c, m, t): return "|".join((cond, c, m, t))


def run_all(args, raw_path):
    gains = load_gains()
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}
    meta = dict(gains=gains.to_dict(), n_eps=args.n_eps)
    if raw.get("_meta") != meta: raw = {"_meta": meta}
    configs, tasks = args.configs or CONFIGS, args.tasks or TASKS
    conds, methods = args.conds or list(CONDS), args.methods or list(METHODS)
    todo = [(cd, c, m, t) for cd in conds for c in configs for t in tasks for m in methods if key(cd, c, m, t) not in raw]
    t_start = time.time()
    for i, (cd, c, m, t) in enumerate(todo):
        t0 = time.time()
        name, ckw = METHODS[m]
        r = raw[key(cd, c, m, t)] = run_cell(ALL_CONFIGS[c], name, t, args.n_eps, gains, perturb=perturb_of(cd, ALL_CONFIGS[c]), ckw=ckw)
        raw_path.write_text(json.dumps(raw))
        eta = (time.time() - t_start) / (i + 1) * (len(todo) - i - 1) / 60
        print(f"[{i + 1}/{len(todo)}] {cd:9s} {c:7s} {m:16s} {t:8s} succ={np.mean(r['success']):.2f} "
              f"score={np.mean(r['score']):.3f} ({time.time() - t0:.0f}s, 剩約 {eta:.0f} 分)", flush=True)
    return raw, gains


# ============================================================================
# 彙整
# ============================================================================
def sc(raw, cd, c, m, t): return float(np.mean(raw[key(cd, c, m, t)]["score"]))
def su(raw, cd, c, m, t): return 100 * float(np.mean(raw[key(cd, c, m, t)]["success"]))
def jp(raw, cd, c, m, t): return float(np.percentile(raw[key(cd, c, m, t)]["jmax"], 95))


def cells_of(scope): return [(c, t) for c in scope for t in TASKS]


def have(raw, cd, scope, methods):
    return all(key(cd, c, m, t) in raw for c, t in cells_of(scope) for m in methods)


def pooled(raw, cd, m, scope, f=sc): return float(np.mean([f(raw, cd, c, m, t) for c, t in cells_of(scope)]))


def degr(raw, cd, m, scope):
    """範圍內平均分數相對自己標稱的退化 pp。"""
    a, b = pooled(raw, cd, m, scope), pooled(raw, "nominal", m, scope)
    return 100 * (1 - a / b) if b > 0.05 else math.nan


def lag_cell(raw, cd, c, m, t, methods):
    best = max(sc(raw, cd, c, x, t) for x in methods)
    return 100 * (1 - sc(raw, cd, c, m, t) / best) if best > 0.05 else math.nan


def lag(raw, cd, m, scope, methods): return float(np.nanmean([lag_cell(raw, cd, c, m, t, methods) for c, t in cells_of(scope)]))


def best_method(raw, cd, scope, methods): return max(methods, key=lambda m: pooled(raw, cd, m, scope))


def table(head, rows): return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])


def main_table(raw, conds, methods):
    out = []
    for cd in conds:
        out.append(f"### {COND_DESC[cd]}（`{cd}`）\n")
        rows = []
        for sname, scope in SCOPES.items():
            if sname.startswith("全部") or not have(raw, cd, scope, methods): continue
            for t in TASKS:
                rows.append([sname.split("（")[0], t] + [f"{np.mean([sc(raw, cd, c, m, t) for c in scope]):.3f} / {np.mean([su(raw, cd, c, m, t) for c in scope]):.0f}% / {np.mean([jp(raw, cd, c, m, t) for c in scope]):.2f}"
                                                       for m in methods])
        out += [table(["範圍", "任務"] + methods, rows), ""]
    return "\n".join(out)


def degr_tables(raw, conds, methods):
    out = []
    for sname, scope in SCOPES.items():
        cs = [cd for cd in conds if cd != "nominal" and have(raw, cd, scope, methods) and have(raw, "nominal", scope, methods)]
        if not cs: continue
        rows = [[sname.split("（")[0], "標稱分數 / 成功率 %"] + [""] * len(cs)] if False else []
        rows = [[m, f"{pooled(raw, 'nominal', m, scope):.3f} / {pooled(raw, 'nominal', m, scope, su):.0f}%"] +
                [f"{degr(raw, cd, m, scope):.1f}（成功率 {pooled(raw, cd, m, scope, su) - pooled(raw, 'nominal', m, scope, su):+.0f}）" for cd in cs] for m in methods]
        out += [f"### {sname}\n", "分數退化 pp（括號：成功率變化 pp，負 = 變差）；退化 = 100(1 − 誤差下平均分數 / 自己標稱平均分數)。\n",
                table(["方法", "標稱 分數 / 成功率"] + [COND_DESC[cd] for cd in cs], rows), ""]
    return "\n".join(out)


def lag_tables(raw, conds, methods):
    out = []
    for sname, scope in SCOPES.items():
        cs = [cd for cd in conds if have(raw, cd, scope, methods)]
        if not cs: continue
        rows = [[m] + [f"{lag(raw, cd, m, scope, methods):.1f}" for cd in cs] for m in methods]
        rows.append(["**該條件最佳方法**"] + [best_method(raw, cd, scope, methods) for cd in cs])
        out += [f"### {sname}\n", table(["方法"] + [COND_DESC[cd] for cd in cs], rows), ""]
    return "\n".join(out)


def appendix(raw, conds, methods):
    out = []
    for cd in conds:
        rows = []
        for c in CONFIGS:
            for t in TASKS:
                if not all(key(cd, c, m, t) in raw for m in methods): continue
                rows.append([c, t] + [f"{sc(raw, cd, c, m, t):.3f} / {su(raw, cd, c, m, t):.0f}% / {jp(raw, cd, c, m, t):.2f}" for m in methods])
        out += [f"### {COND_DESC[cd]}\n", table(["構型", "任務"] + methods, rows), ""]
    return "\n".join(out)


def verdict(raw, conds, methods):
    """自動算出的判定數字（解讀在 g2_notes.md，人工撰寫）。"""
    scope = TRAIN_G + DISC                                  # 判定用 5 構型（TRAIN + 鑑別組）；對照組 grid9 另列
    errs = [cd for cd in conds if cd != "nominal" and have(raw, cd, scope, methods) and have(raw, "nominal", scope, methods)]
    if not errs: return "（資料不足）"
    cls = [m for m in methods if m != "B5"]
    L = []
    # Q1：最強古典法 = 每個條件在 5 構型 × 2 任務上平均分數最高的方法（B5 是天真基準，不算）；對照「標稱下最強古典法分數」
    nom_best = best_method(raw, "nominal", scope, cls)
    nom_best_score = pooled(raw, "nominal", nom_best, scope)
    rows = []
    for cd in errs:
        bm = best_method(raw, cd, scope, cls)
        s = pooled(raw, cd, bm, scope)
        rows.append([COND_DESC[cd], bm, f"{s:.3f}", f"{pooled(raw, cd, bm, scope, su):.0f}%", f"{100 * (1 - s / nom_best_score):.1f}",
                     f"{degr(raw, cd, bm, scope):.1f}", f"{degr(raw, cd, nom_best, scope):.1f}"])
    L.append(f"**Q1 最強古典法在誤差下掉多少**（TRAIN + 鑑別組 5 構型 × 2 任務；每個條件取平均分數最高的古典法；"
             f"標稱下最強是 {nom_best}，分數 {nom_best_score:.3f}）\n")
    L.append(table(["條件", "該條件最強古典法", "分數", "成功率", "相對『標稱最強』掉 pp", "它自己的退化 pp", f"{nom_best} 自己的退化 pp"], rows))
    # Q2：哪種誤差最傷（所有古典法的退化平均、與最強古典法的退化）
    rows = []
    for cd in errs:
        rows.append([COND_DESC[cd], f"{np.nanmean([degr(raw, cd, m, scope) for m in cls]):.1f}",
                     f"{np.nanmax([degr(raw, cd, m, scope) for m in cls]):.1f}", f"{np.nanmin([degr(raw, cd, m, scope) for m in cls]):.1f}",
                     f"{np.nanmean([pooled(raw, cd, m, scope, su) - pooled(raw, 'nominal', m, scope, su) for m in cls]):+.1f}"])
    L.append("\n**Q2 哪種誤差最傷**（7 個古典方法各自的分數退化 pp 的平均／最大／最小；最後一欄是成功率變化 pp 的平均）\n")
    L.append(table(["條件", "平均退化", "最大", "最小", "成功率變化"], rows))
    # Q3：利基 = 最強古典法離「標稱天花板」（每格標稱下最佳古典法分數）的落後，逐 (條件, 構型, 任務) 格
    rows, niche = [], []
    for cd in errs:
        for c in CONFIGS:
            vals = []
            for t in TASKS:
                ceil = max(sc(raw, "nominal", c, m, t) for m in cls)
                bm = max(sc(raw, cd, c, m, t) for m in cls)
                vals.append(100 * (1 - bm / ceil) if ceil > 0.05 else math.nan)
            niche.append((cd, c, *vals))
    L.append("\n**Q3 RL 的利基落在哪**：每格的「最強古典法 vs 標稱天花板」落後 pp = 100(1 − 該條件下最佳古典法分數 / 標稱下最佳古典法分數)。"
             "天花板假設一個無需辨識、完全魯棒的控制器至少能回到標稱水準（對 RL 是樂觀上界，真正的 RL 不一定做得到）。≥ 5 pp 的格：\n")
    rows = [[COND_DESC[cd], c] + [fmt(v, 1) for v in vs] for cd, c, *vs in niche if any(v >= 5 for v in vs if not math.isnan(v))]
    L.append(table(["條件", "構型"] + TASKS, rows) if rows else "（沒有任何格 ≥ 5 pp）")
    n_tot = len(niche) * len(TASKS)
    n_5 = sum(v >= 5 for _, _, *vs in niche for v in vs if not math.isnan(v))
    L.append(f"\n共 {n_5} / {n_tot} 格（條件 × 構型 × 任務）≥ 5 pp。")
    return "\n".join(L)


def feasibility_table():
    """物理可行性：payload 下，整架能不能懸停（sum T = Mg、三軸力矩 = 0、0 ≤ T ≤ Tmax 有解？），
    以及懸停時 roll／pitch 還剩多少力矩權限（線性規劃：在懸停等式下，最大化該軸 ±力矩；取兩個方向較小者；與標稱比）。"""
    from scipy.optimize import linprog
    from quadsim.env import ModularEnv
    rows = []
    for c in CONFIGS:
        cells = ALL_CONFIGS[c]
        host = int(np.argmax([x[0] for x in cells]))
        res = {}
        for frac in (0.0, 0.10, 0.15, 0.30):
            env = ModularEnv(cells, 1, task="POS_STEP", perturb=Perturb(payload_frac=frac, payload_host=host))
            g = env.geo; n = 4 * env.N; Mg = g["M"][0].item() * 9.81
            tdir, tq = g["tdir"][0, :, 2].numpy(), g["tq"][0].numpy()
            A = np.vstack([tdir, tq.T])
            ok = linprog(np.zeros(n), A_eq=A, b_eq=[Mg, 0, 0, 0], bounds=[(0, env.tmax)] * n).status == 0
            auth = []
            for ax in (0, 1):
                v = []
                for sg in (1, -1):
                    r = linprog(-sg * tq[:, ax], A_eq=np.delete(A, 1 + ax, 0), b_eq=[Mg, 0, 0] if ax == 0 else [Mg, 0, 0], bounds=[(0, env.tmax)] * n)
                    v.append(-r.fun * sg if r.status == 0 else math.nan)
                auth.append(min(v))
            res[frac] = (ok, auth, n * env.tmax / Mg)
        f = lambda frac: ("可" if res[frac][0] else "**不可**") + (f"（TWR {res[frac][2]:.2f}；roll {res[frac][1][0] / res[0.0][1][0]:.2f}、pitch {res[frac][1][1] / res[0.0][1][1]:.2f}）"
                                                                if res[frac][0] else f"（TWR {res[frac][2]:.2f}）")
        rows.append([c, f(0.10), f(0.15), f(0.30)])
    return table(["構型", "payload 10%", "payload 15%", "payload 30%"], rows)


def write_report(raw, gains, args, path):
    conds = [cd for cd in CONDS if any(k.startswith(cd + "|") for k in raw)]
    methods = [m for m in METHODS if any(f"|{m}|" in k for k in raw)]
    g = gains.to_dict()
    notes = RES / f"g2_notes{args.tag}.md"
    md = [f"# G2：古典控制在物理誤差下的退化（{time.strftime('%Y-%m-%d')}）", "",
          f"由 `scripts/g2_mismatch.py` 產生（原始資料 `g2_raw{args.tag}.json`）。每格 {raw['_meta']['n_eps']} 個 episode，固定 seed；"
          "同一構型、任務、條件下所有方法的初始條件、第一個 episode 的幾何與 T2 目標序列相同（成對）。"
          f"外迴路增益同 g1_v2（kp={g['kp']:.2f}、kd={g['kd']:.2f}、kR={g['kR']:.1f}、kw={g['kw']:.1f}、yaw 倍率 {g['yaw_ratio']:.2f}），所有方法共用，沒有為誤差重調。"
          "所有方法只拿到標稱形狀與質量。", "",
          "構型：TRAIN line3、square4；鑑別組 line6、line8、grid10；對照組 grid9。任務 POS_STEP、T2。"
          "表中格式「分數 / 成功率 / 接頭負載比 p95」；範圍內的構型先各自算再平均。", "",
          "## 物理可行性（payload 掛在 x 最大端；線性規劃）", "",
          "懸停是否有可行的推力分配；括號：整架推重比、懸停下 roll／pitch 的剩餘力矩權限（相對無 payload 的同構型，取 ± 兩個方向較小者）。"
          "**不可** = 連懸停都不可能，任何控制器（包含 RL）都會失敗。\n", feasibility_table(), "",
          "## 判定（自動計算的數字）", "", verdict(raw, conds, methods), "",
          notes.read_text() if notes.exists() else "（尚無文字解讀）", "",
          "## 主表（各條件、各方法）", "", main_table(raw, conds, methods),
          "## 退化表：相對自己在標稱物理下", "", degr_tables(raw, conds, methods),
          "## 落後表：相對同一誤差條件下的最佳方法（含 B5）", "",
          "每格 100(1 − 分數 / 該格最佳方法分數)，再對範圍內的格子平均。\n", lag_tables(raw, conds, methods),
          "## 附錄：逐格（分數 / 成功率 / 接頭負載比 p95）", "", appendix(raw, conds, methods)]
    path.write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eps", type=int, default=100)
    ap.add_argument("--tag", default="")
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--conds", nargs="*")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    raw_path = RES / f"g2_raw{args.tag}.json"
    if args.report_only: raw, gains = json.loads(raw_path.read_text()), load_gains()
    else: raw, gains = run_all(args, raw_path)
    write_report(raw, gains, args, RES / f"g2{args.tag}.md")
    print("已寫入", RES / f"g2{args.tag}.md")


if __name__ == "__main__":
    main()
