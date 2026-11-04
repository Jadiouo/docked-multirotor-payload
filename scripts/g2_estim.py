"""G2-estim：「估計式 informed」古典對手（B1-QP + 線上估計 payload 質量與重心）能補回多少「不知道 payload」的缺口？

對手 B1-QP-estim（quadsim/controllers.py::B1QPEstim）：只用陀螺儀、加速度計與自己送出的馬達指令（經標稱一階馬達模型），
  帶遺忘 RLS 估總質量 M̂ 與重心偏移 ĉ_xy，把 payload 當質點重建 CoM、J、A，餵進和 B1-QP-informed 相同的分配器。
超參數（公平性）：在驗證 seed（env seed = 100 + 任務 seed，與評估用的 0/1/2 不重疊）上，line6、line8 × {T2, POS_STEP, ATT_STEP} 的 payload15
  做 20 組隨機搜尋（第 0 組 = 預設值，其餘 19 組隨機，種子寫死），取驗證分數最高的一組；yawlim 版本沿用同一組超參數，不另調。
正式評估：照 g2_informed.py（payload15、掛在 x 最大的模組、同 seed、100 episode），6 構型 × {T2, POS_STEP, ATT_STEP}。
最強非 informed 古典法 C = 該格 {B1-QP, B1-QP-yawlim, B3 INDI, B3+adapt} 中平均分數最高者（逐格挑最大，對古典法有利）。
補回比例 = (E − C)/(I − C)，E = 估計式（含 yawlim 時對照 informed+yawlim），I = informed；只在缺口 100(1 − C/I) ≥ 5 pp 的格算。
留出評估：payload 質量 {0.10, 0.20} × 隨機掛點（固定 seed，每組 3 個掛點、各 40 episode 合併），line6、line8 的 T2。
yawlim 版本：gains.yaw_ratio 與 rate_c 取自 results/p2/g1_coord_tune.json 的 best（同 g1_coord.py）。
執行：CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \\
        .venv/bin/python -m scripts.g2_estim --stage tune|eval|holdout|report [--n-eps 100] [--tag ""]"""
import argparse
import dataclasses
import json
import math
import time
import numpy as np

import scripts.g1_gate as g1
from scripts.g1_gate import RES, ALL_CONFIGS, run_cell, load_gains
from scripts.g2_mismatch import perturb_of, table
from quadsim.env import Perturb

EVAL_CONFIGS = ["line6", "line7", "line8", "grid10", "line3", "square4"]
TASKS = ["T2", "POS_STEP", "ATT_STEP"]
CLASSICAL = ["B1-QP", "B1-QP-yawlim", "B3 INDI", "B3+adapt"]
ESTIM = ["B1-QP-estim", "B1-QP-estim-yawlim"]
INFORMED = ["B1-QP-informed", "B1-QP-informed-yawlim"]
METHODS = CLASSICAL + INFORMED + ESTIM
VAL_OFF, VAL_CONFIGS, VAL_TASKS, VAL_EPS = 100, ["line6", "line8"], ["T2", "POS_STEP", "ATT_STEP"], 32
TUNE_SEED, N_TRIALS = 20261030, 20
HOLD_MASS, HOLD_CONFIGS, HOLD_HOSTS, HOLD_EPS, HOLD_SEED = [0.10, 0.20], ["line6", "line8"], 3, 40, 12345
DEFAULT = dict(warm_c=3, warm_m=10, lam_m=0.99, eps_m=0.1, lam_c=0.995, eps_c=0.05, snap=False, corr=True)


def space_sample(rng):
    return dict(warm_c=int(rng.choice([1, 3, 6, 10])), warm_m=int(rng.choice([3, 6, 10, 20])), lam_m=float(rng.uniform(0.97, 0.999)),
                eps_m=float(np.exp(rng.uniform(np.log(0.01), np.log(1.0)))), lam_c=float(rng.uniform(0.98, 0.999)),
                eps_c=float(np.exp(rng.uniform(np.log(0.005), np.log(0.5)))), snap=bool(rng.integers(2)), corr=bool(rng.integers(2)))


def setup(method, gains):
    """label → (gains, ckw)；yawlim 版本用 g1_coord_tune 的 best。"""
    if "yawlim" not in method: return gains, {}
    b = json.loads((RES / "g1_coord_tune.json").read_text())["best"]
    return dataclasses.replace(gains, yaw_ratio=b["yaw_ratio"]), dict(rate_c=b["rate_c"])


def run(cells, method, task, n, gains, pt=None, est=None, seed_off=0, pay=None):
    """跑一格。pt：Perturb；pay：(payload_frac, host)（informed 要知道）；est：估計式超參數。seed_off：env seed 位移（驗證用）。"""
    g, ckw = setup(method, gains)
    if pt is not None and pt.payload_frac > 0 and "informed" in method:
        ckw = {**ckw, "payload_frac": pt.payload_frac, "payload_host": pt.payload_host}
    if "estim" in method: ckw = {**ckw, **(est or DEFAULT)}
    old = dict(g1.SEED)
    try:
        g1.SEED.update({t: old[t] + seed_off for t in old})
        return run_cell(cells, method, task, n, g, perturb=pt, ckw=ckw)
    finally:
        g1.SEED.update(old)


def load(path, meta):
    raw = json.loads(path.read_text()) if path.exists() else {}
    if raw.get("_meta") != meta: raw = {"_meta": meta}
    return raw


def stage_tune(args):
    gains, path = load_gains(), RES / f"g2_estim_tune{args.tag}.json"
    rng = np.random.default_rng(TUNE_SEED)
    trials = [dict(DEFAULT, note="預設值")] + [dict(space_sample(rng), note="") for _ in range(N_TRIALS - 1)]
    n = args.n_eps if args.n_eps != 100 else VAL_EPS
    out = json.loads(path.read_text()) if path.exists() else {}
    res = out.get("trials", []) if out.get("n_eps") == n else []
    ref = {}
    for i, tr in enumerate(trials):
        if i < len(res): continue
        t0 = time.time()
        est = {k: v for k, v in tr.items() if k != "note"}
        sc = {}
        for c in VAL_CONFIGS:
            pt = perturb_of("payload15", ALL_CONFIGS[c])
            for t in VAL_TASKS:
                sc[f"{c}|{t}"] = float(np.mean(run(ALL_CONFIGS[c], "B1-QP-estim", t, n, gains, pt, est, VAL_OFF)["score"]))
        score = float(np.mean(list(sc.values())))
        res.append(dict(tr, score=score, cells=sc))
        path.write_text(json.dumps(dict(trials=res, n_eps=n, val_seed_offset=VAL_OFF), indent=1))
        print(f"[tune {i}] score={score:.4f} {({k: round(v, 3) if isinstance(v, float) else v for k, v in est.items()})} ({time.time() - t0:.0f}s)", flush=True)
    best = max(res, key=lambda r: r["score"])
    # 參考：同驗證 seed 上的 B1-QP-informed 與 B3+adapt
    for m in ["B1-QP-informed", "B3+adapt", "B1-QP"]:
        ref[m] = float(np.mean([np.mean(run(ALL_CONFIGS[c], m, t, n, gains, perturb_of("payload15", ALL_CONFIGS[c]), None, VAL_OFF)["score"])
                                for c in VAL_CONFIGS for t in VAL_TASKS]))
    path.write_text(json.dumps(dict(trials=res, best=best, reference_val_scores=ref, n_eps=n, val_seed_offset=VAL_OFF,
                                    configs=VAL_CONFIGS, tasks=VAL_TASKS), indent=1))
    print("best", best, "ref", ref)


def best_est(args):
    f = RES / f"g2_estim_tune{args.tag}.json"
    b = json.loads(f.read_text())["best"]
    return {k: b[k] for k in DEFAULT}


def stage_eval(args):
    gains, est = load_gains(), best_est(args)
    path = RES / f"g2_estim_raw{args.tag}.json"
    raw = load(path, dict(n_eps=args.n_eps, est=est, gains=gains.to_dict()))
    todo = [(c, t, m) for c in EVAL_CONFIGS for t in TASKS for m in METHODS if f"eval|{c}|{t}|{m}" not in raw]
    t_start = time.time()
    for i, (c, t, m) in enumerate(todo):
        t0 = time.time()
        r = raw[f"eval|{c}|{t}|{m}"] = run(ALL_CONFIGS[c], m, t, args.n_eps, gains, perturb_of("payload15", ALL_CONFIGS[c]), est)
        path.write_text(json.dumps(raw))
        eta = (time.time() - t_start) / (i + 1) * (len(todo) - i - 1) / 60
        print(f"[{i + 1}/{len(todo)}] {c:7s} {t:8s} {m:22s} succ={np.mean(r['success']):.2f} score={np.mean(r['score']):.3f} ({time.time() - t0:.0f}s, 剩約 {eta:.0f} 分)", flush=True)


def hold_hosts(c, frac):
    rng = np.random.default_rng([HOLD_SEED, int(frac * 100), len(ALL_CONFIGS[c])])
    return sorted(int(h) for h in rng.choice(len(ALL_CONFIGS[c]), HOLD_HOSTS, replace=False))


def stage_holdout(args):
    gains, est = load_gains(), best_est(args)
    path = RES / f"g2_estim_raw{args.tag}.json"
    raw = load(path, dict(n_eps=args.n_eps, est=est, gains=gains.to_dict()))
    n = HOLD_EPS if args.n_eps == 100 else args.n_eps
    todo = [(c, fr, h, m) for c in HOLD_CONFIGS for fr in HOLD_MASS for h in hold_hosts(c, fr) for m in METHODS if f"hold|{c}|{fr}|{h}|{m}" not in raw]
    t_start = time.time()
    for i, (c, fr, h, m) in enumerate(todo):
        t0 = time.time()
        r = raw[f"hold|{c}|{fr}|{h}|{m}"] = run(ALL_CONFIGS[c], m, "T2", n, gains, Perturb(payload_frac=fr, payload_host=h), est)
        path.write_text(json.dumps(raw))
        eta = (time.time() - t_start) / (i + 1) * (len(todo) - i - 1) / 60
        print(f"[{i + 1}/{len(todo)}] hold {c} m={fr} host={h} {m:22s} succ={np.mean(r['success']):.2f} score={np.mean(r['score']):.3f} ({time.time() - t0:.0f}s, 剩約 {eta:.0f} 分)", flush=True)


# ============================================================================
# 報告
# ============================================================================
def S(raw, k): return np.array(raw[k]["score"], float)
def U(raw, k): return np.array(raw[k]["success"], float)
def f3(x): return f"{x:.3f}"
def cellstr(raw, k): return f"{S(raw, k).mean():.3f} / {100 * U(raw, k).mean():.0f}%"


def boot(fn, arrs, rng, nb=2000):
    """arrs：同長度 episode 陣列的 list；fn(*各陣列重抽後的平均) → 統計量。回傳 (點估計, 下, 上)。"""
    n = len(arrs[0])
    idx = rng.integers(0, n, (nb, n))
    with np.errstate(all="ignore"): b = fn(*[a[idx].mean(1) for a in arrs])
    return float(fn(*[a.mean() for a in arrs])), *[float(x) for x in np.nanpercentile(b, [2.5, 97.5])]


def ci(t, d=3): return "—" if t is None or math.isnan(t[0]) else f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"


def best_c(raw, pre):
    return max(CLASSICAL, key=lambda m: S(raw, f"{pre}|{m}").mean())


def analyze(raw, pre, rng, Ekey):
    """pre：不含方法名的前綴。回傳該格的各量。Ekey：估計式名稱（對照的 informed 依是否 yawlim 決定）。"""
    ik = "B1-QP-informed-yawlim" if "yawlim" in Ekey else "B1-QP-informed"
    cm = best_c(raw, pre)
    C, I, E = S(raw, f"{pre}|{cm}"), S(raw, f"{pre}|{ik}"), S(raw, f"{pre}|{Ekey}")
    gap = 100 * (1 - C.mean() / I.mean()) if I.mean() > 0.05 else math.nan
    d = dict(cm=cm, C=C.mean(), I=I.mean(), E=E.mean(), gap=gap,
             EC=boot(lambda e, c: e - c, [E, C], rng), IE=boot(lambda i, e: i - e, [I, E], rng))
    d["rec"] = boot(lambda e, c, i: (e - c) / (i - c), [E, C, I], rng) if gap >= 5 else None
    return d


def report(args):
    raw = json.loads((RES / f"g2_estim_raw{args.tag}.json").read_text())
    tune = json.loads((RES / f"g2_estim_tune{args.tag}.json").read_text())
    rng = np.random.default_rng(0)
    est = raw["_meta"]["est"]
    L = [f"# G2-estim：估計式 informed 古典對手（{time.strftime('%Y-%m-%d')}）", "",
         "由 `scripts/g2_estim.py` 產生；原始資料 `g2_estim_raw.json`、調參紀錄 `g2_estim_tune.json`。方法定義見 `quadsim/controllers.py::B1QPEstim`。", "",
         "**B1-QP-estim**：B1-QP 的分配；質量與重心由線上估計取代標稱值。只用陀螺儀（含雜訊）、加速度計（含雜訊）與自己送出的馬達指令（經標稱一階馬達模型）。"
         "總質量 M̂：加速度計 z 比力平均 vs ΣT 的帶遺忘 RLS；重心偏移 ĉ_xy：力矩殘差 Jα+ω×Jω−τ_c 對 ΣT 的帶遺忘 RLS；"
         "再把 payload 當質點（m̂_p = M̂ − M_nom，p̂ = ĉ M̂/m̂_p）重算 CoM、J、A。前 `warm_c`/`warm_m` 步分別用標稱 ĉ/M̂；M̂ ∈ [M_nom, 1.6 M_nom]；p̂ 限制在構型外圍內；"
         "每個 episode 開頭重置。外迴路 α 限幅仍用標稱（同 informed）。**-yawlim** = 加 YawLimMixin（rate_c、yaw_ratio 取自 g1_coord_tune.json 的 best）。", "",
         f"**調參**：驗證 env seed = 100 + 任務 seed（評估用 0/1/2，不重疊），line6、line8 × {{T2, POS_STEP, ATT_STEP}} 的 payload15，每格 {tune['n_eps']} episode；"
         f"20 組（第 0 組預設值、19 組隨機，種子 {TUNE_SEED}）；選中第 {tune['trials'].index(tune['best'])} 組：{ {k: (round(v, 4) if isinstance(v, float) else v) for k, v in est.items()} }，"
         f"驗證分數 {tune['best']['score']:.4f}（預設值 {tune['trials'][0]['score']:.4f}；同 seed 參考：{', '.join(f'{k} {v:.4f}' for k, v in tune['reference_val_scores'].items())}）。yawlim 版本沿用同一組超參數。", "",
         "**最強非 informed 古典法 C** = 該格 {B1-QP, B1-QP-yawlim, B3 INDI, B3+adapt} 中平均分數最高者（逐格挑，對古典法有利）。缺口 = 100(1 − C/I) pp；"
         "補回比例 = (E − C)/(I − C)，只在缺口 ≥ 5 pp 的格算；E−C、I−E 為分數差（成對 bootstrap 2000 次 95% CI，episode 重抽）；補回比例的 CI 是比值的 bootstrap（缺口小時不穩）。", ""]
    sections = {}
    for Ekey in ESTIM:
        ik = "B1-QP-informed-yawlim" if "yawlim" in Ekey else "B1-QP-informed"
        L += [f"## 正式評估（payload15）：{Ekey} 對照 {ik}", "",]
        rows, recs = [], {}
        for c in EVAL_CONFIGS:
            for t in TASKS:
                pre = f"eval|{c}|{t}"
                a = analyze(raw, pre, rng, Ekey)
                recs[(c, t)] = a
                cm = a["cm"]
                rows.append([c, t, cellstr(raw, pre + "|" + cm) + " (" + cm + ")", cellstr(raw, f"{pre}|{ik}"), cellstr(raw, f"{pre}|{Ekey}"),
                             ("%.1f" % a["gap"]) if not math.isnan(a["gap"]) else "—", ci(a["EC"]), ci(a["IE"]),
                             ("%.2f [%.2f, %.2f]" % a["rec"]) if a["rec"] else "—"])
        L += [table(["構型", "任務", "最強非 informed 古典 C（分數/成功率）", "informed I", "估計式 E", "缺口 pp", "E−C [CI]", "I−E [CI]", "補回比例 [CI]"], rows), ""]
        sections[Ekey] = recs
    # 全部方法一覽
    L += ["## 全部方法一覽（payload15，分數 / 成功率）", ""]
    rows = [[c, t] + [cellstr(raw, f"eval|{c}|{t}|{m}") for m in METHODS] for c in EVAL_CONFIGS for t in TASKS]
    L += [table(["構型", "任務"] + METHODS, rows), ""]
    # 彙總
    L += ["## 彙總（缺口 ≥ 5 pp 的格）", ""]
    rows = []
    for Ekey in ESTIM:
        big = [(k, a) for k, a in sections[Ekey].items() if a["gap"] >= 5]
        if not big: rows.append([Ekey, "0", "—", "—", "—"]); continue
        r = np.mean([(a["E"] - a["C"]) / (a["I"] - a["C"]) for _, a in big])
        rows.append([Ekey, str(len(big)), "、".join(f"{c}/{t}" for (c, t), _ in big),
                     f"{r:.2f}", f"{min((a['E'] - a['C']) / (a['I'] - a['C']) for _, a in big):.2f} ~ {max((a['E'] - a['C']) / (a['I'] - a['C']) for _, a in big):.2f}"])
    L += [table(["估計式", "缺口≥5pp 格數", "哪些格", "補回比例（格平均）", "範圍"], rows), ""]
    # 留出
    hold = [k for k in raw if k.startswith("hold|")]
    if hold:
        L += ["## 留出評估：payload 質量 {0.10, 0.20} × 隨機掛點（T2）", "",
              f"每個 (構型, 質量) 3 個隨機掛點（rng 種子 {HOLD_SEED}），每掛點 {HOLD_EPS} episode，合併成 {3 * HOLD_EPS} 個 episode 做成對 bootstrap；估計式超參數沿用調參結果。", ""]
        for Ekey in ESTIM:
            ik = "B1-QP-informed-yawlim" if "yawlim" in Ekey else "B1-QP-informed"
            rows = []
            for c in HOLD_CONFIGS:
                for fr in HOLD_MASS:
                    hs = hold_hosts(c, fr)
                    cat = lambda m: {"score": sum((raw[f"hold|{c}|{fr}|{h}|{m}"]["score"] for h in hs), []),
                                     "success": sum((raw[f"hold|{c}|{fr}|{h}|{m}"]["success"] for h in hs), [])}
                    tmp = {f"x|{m}": cat(m) for m in METHODS}
                    a = analyze(tmp, "x", rng, Ekey)
                    cm = a["cm"]
                    rows.append([c, str(fr), str(hs), cellstr(tmp, "x|" + cm) + " (" + cm + ")", cellstr(tmp, f"x|{ik}"), cellstr(tmp, f"x|{Ekey}"),
                                 ("%.1f" % a["gap"]) if not math.isnan(a["gap"]) else "—", ci(a["EC"]), ci(a["IE"]), ("%.2f [%.2f, %.2f]" % a["rec"]) if a["rec"] else "—"])
            L += [f"### {Ekey} 對照 {ik}", "", table(["構型", "質量比", "掛點", "C", "informed I", "估計式 E", "缺口 pp", "E−C [CI]", "I−E [CI]", "補回比例 [CI]"], rows), ""]
    # 事實陳述
    L += ["## 事實陳述（不下結論）", ""]
    for Ekey in ESTIM:
        recs = sections[Ekey]
        big = [(k, a) for k, a in recs.items() if a["gap"] >= 5]
        better = [(k, a) for k, a in recs.items() if a["EC"][1] > 0]
        worse = [(k, a) for k, a in recs.items() if a["EC"][2] < 0]
        L.append(f"- {Ekey}：18 格中缺口 ≥ 5 pp 的有 {len(big)} 格；E−C 的 CI 全為正的有 {len(better)} 格，全為負的有 {len(worse)} 格"
                 + ("；缺口格的補回比例（逐格）：" + "、".join("%s/%s %.2f" % (c, t, (a["E"] - a["C"]) / (a["I"] - a["C"])) for (c, t), a in big) + "。" if big else "。"))
    if hold:
        parts = []
        for Ekey in ESTIM:
            ik = "B1-QP-informed-yawlim" if "yawlim" in Ekey else "B1-QP-informed"
            gaps = []
            for c in HOLD_CONFIGS:
                for fr in HOLD_MASS:
                    hs = hold_hosts(c, fr)
                    tmp = {f"x|{m}": {"score": sum((raw[f"hold|{c}|{fr}|{h}|{m}"]["score"] for h in hs), []), "success": []} for m in METHODS}
                    gaps.append(S(tmp, "x|" + Ekey).mean() - S(tmp, "x|" + ik).mean())
            parts.append("%s 減 %s 的分數差：%s" % (Ekey, ik, "、".join("%+.3f" % g for g in gaps)))
        L.append("- 留出評估（line6/0.1、line6/0.2、line8/0.1、line8/0.2，T2）：" + "；".join(parts) + "。")
    L.append(f"- 調參：20 組驗證分數 {min(t['score'] for t in tune['trials']):.4f} ~ {max(t['score'] for t in tune['trials']):.4f}，預設值 {tune['trials'][0]['score']:.4f}，選中者 {tune['best']['score']:.4f}"
             f"（snap={est['snap']}：把 payload 位置吸附到最近的模組中心，等於假設 payload 掛在模組中心；正式評估與留出評估的掛點都在模組中心，沒有測偏離中心的掛點）。")
    (RES / f"g2_estim{args.tag}.md").write_text("\n".join(L) + "\n")
    print("已寫入", RES / f"g2_estim{args.tag}.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tune", "eval", "holdout", "report"])
    ap.add_argument("--n-eps", type=int, default=100)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    {"tune": stage_tune, "eval": stage_eval, "holdout": stage_holdout, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
