"""古典控制器 confirmatory seed 重算（prereg §14(a)；tag prereg-v1.0 之後的確認性報告）。

不改任何規則、構型、任務、比較對象或控制器參數；只把環境 seed 換成 confirmatory（offset 900000，prereg 3.1）。
全部重用既有函式，不重寫動力學／控制器：
  E1 / E2 / E3 / 6.1 標稱：scripts.p3d_factorial.run_one（與 rl_eval.compare_e3、estim_cell 同一條路徑；
      參數 = g2_estim_tune.json best + g1_coord_tune.json best，estim 一律 snap=False），搭配 rl_eval.seed_shift 位移 g1_gate.SEED。
  RQ-D event：scripts.rqd_classical.run_ev（seed = SEED_EV[task] + offset；偵測超參數 = rqd_tune.json best）。
涵蓋的格（line6/7/8 為主；event 含 grid10）：
  E1   payload15（argmax(x) 模組中心、t=0）x line6/7/8 x {T2, POS_STEP} x 9 個方法（POS_STEP 為次要，prereg 5.0）
  E2   留出集（質量 {.10,.15,.20} x 每 episode 隨機掛點）x line6/7/8 x {T2, POS_STEP} x {B1, B1-QP, B3 INDI, B3+adapt, estim-yawlim}
  E3   偏心 payload15（p3d_factorial.OFFSET）x line6/7/8 x {POS_STEP, T2} x {B1-QP, B1-QP-yawlim, estim-yawlim, informed-offset-yawlim}，
       另算 T2+POS_STEP 合併（逐 episode 平均，與 rl_eval 的 (T2+POS)/2 同）
  NOM  6.1 標稱（無 payload）x line6/7/8 x {T2, POS_STEP} x {B1, B1-QP, B1-QP-yawlim, estim-yawlim}
  EV   RQ-D：{line6,line7,line8,grid10} x {T2,POS_STEP,HOVER} x {attach,drop}x{center,offcenter} x 4 個古典方法（48 格）
階段：
  run     跑格子，增量寫入 --scratch/<group>.json（可續跑；n_eps／seed 不符就重來）。不同 group 可同時用不同行程跑。
  report  把 --scratch 的各 group 合併成 <out-dir>/classical_confirm_raw.json 與 classical_confirm.md。
  預設（--stage all）= run 然後 report。
用法（專案根目錄；CPU）：
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. \\
     .venv/bin/python -m scripts.classical_confirm [--only E1 E2 E3 NOM EV] [--configs ..] [--tasks ..] [--methods ..] [--n-eps 100]
"""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import scripts.g1_gate as g1
from scripts.g1_gate import RES, ALL_CONFIGS, load_gains
from scripts.rl_eval import seed_shift
from scripts import p3d_factorial as p3
from scripts import rqd_classical as rq

ROOT = Path(__file__).resolve().parent.parent
CONF_OFF = 900000
SIGN_EPS = 0.002                       # 方向判定：|差| < 0.002 視為 ≈0
BOOT_N, BOOT_SEED = 2000, 0

# 方法註冊：label -> (run_one 的控制器名, nosnap, informed, 探索 seed 舊值用的 p3d 標籤)
M = {
    "B1": ("B1", False, False, "000"),
    "B1-QP": ("B1-QP", False, False, "100"),
    "B1-QP-yawlim": ("B1-QP-yawlim", False, False, "110"),
    "B1-QP-estim(snap=False)": ("B1-QP-estim", True, False, "101n"),
    "B1-QP-estim-yawlim(snap=False)": ("B1-QP-estim-yawlim", True, False, "111n"),
    "B3 INDI": ("B3 INDI", False, False, None),
    "B3+adapt": ("B3+adapt", False, False, None),
    "B1-QP-informed": ("B1-QP-informed", False, True, None),
    "B1-QP-informed-yawlim": ("B1-QP-informed-yawlim", False, True, "informed"),
    "B1-QP-informed-offset-yawlim": ("B1-QP-informed-offset-yawlim", False, True, "informed"),
}
EST_Y, EST, INF_Y = "B1-QP-estim-yawlim(snap=False)", "B1-QP-estim(snap=False)", "B1-QP-informed-yawlim"
INF_OFF = "B1-QP-informed-offset-yawlim"
CFG3 = ["line6", "line7", "line8"]
GROUPS = {
    "E1": dict(cond="payload15", cfgs=CFG3, tasks=["T2", "POS_STEP"],
               methods=["B1", "B1-QP", "B1-QP-yawlim", "B3 INDI", "B3+adapt", "B1-QP-informed", INF_Y, EST, EST_Y],
               contrasts=[(INF_Y, EST_Y), (EST_Y, EST), (EST_Y, "B1-QP-yawlim")]),
    "E2": dict(cond="E2", cfgs=CFG3, tasks=["T2", "POS_STEP"], fracs=p3.E2_FRACS,
               methods=["B1", "B1-QP", "B3 INDI", "B3+adapt", EST_Y],
               contrasts=[(EST_Y, "B1-QP"), (EST_Y, "B3 INDI"), (EST_Y, "B3+adapt")]),
    "E3": dict(cond="offcenter", cfgs=CFG3, tasks=["POS_STEP", "T2"],
               methods=["B1-QP", "B1-QP-yawlim", EST_Y, INF_OFF],
               contrasts=[(INF_OFF, EST_Y), (EST_Y, "B1-QP-yawlim")]),
    "NOM": dict(cond="nominal", cfgs=CFG3, tasks=["T2", "POS_STEP"],
                methods=["B1", "B1-QP", "B1-QP-yawlim", EST_Y], contrasts=[]),
    "EV": dict(cfgs=rq.CFGS, tasks=rq.TASKS, conds=rq.CONDS, methods=rq.METHODS,
               contrasts=[("informed-event-yawlim", "estim-reset-yawlim"), ("estim-reset-yawlim", "estim-yawlim"),
                          ("estim-yawlim", "B1-QP-yawlim")]),
}
ENV_DESC = {"E1": "payload15（15%，argmax(x) 模組中心，t=0）", "E2": "留出集 RandomPayloadEnv（質量固定 x 每 episode 隨機掛點）",
            "E3": "offcenter（payload15 + p3d_factorial.OFFSET）", "NOM": "nominal（無 payload）",
            "EV": "中途 payload 事件（rqd_classical）"}
E3_PREREG = {  # prereg 5.b（estim / informed），confirmatory seed 900000
    ("line6", "POS_STEP", EST_Y): 0.548, ("line6", "POS_STEP", INF_OFF): 0.595,
    ("line7", "POS_STEP", EST_Y): 0.498, ("line7", "POS_STEP", INF_OFF): 0.559,
    ("line8", "POS_STEP", EST_Y): 0.385, ("line8", "POS_STEP", INF_OFF): 0.494,
    ("line6", "T2", EST_Y): 0.210, ("line6", "T2", INF_OFF): 0.209,
    ("line7", "T2", EST_Y): 0.182, ("line7", "T2", INF_OFF): 0.181,
    ("line8", "T2", EST_Y): 0.163, ("line8", "T2", INF_OFF): 0.168,
}


# ---------------------------------------------------------------------------- 工作清單
def jobs(args):
    out = []
    for g, S in GROUPS.items():
        if args.only and g not in args.only: continue
        for cfg in S["cfgs"]:
            if args.configs and cfg not in args.configs: continue
            for task in S["tasks"]:
                if args.tasks and task not in args.tasks: continue
                if g == "EV":
                    for typ, off in S["conds"]:
                        for m in S["methods"]:
                            if args.methods and m not in args.methods: continue
                            out.append((g, f"{cfg}|{task}|{typ}-{off}|{m}", dict(cfg=cfg, task=task, typ=typ, off=off, method=m)))
                else:
                    for fr in S.get("fracs", (None,)):
                        for m in S["methods"]:
                            if args.methods and m not in args.methods and short(m) not in args.methods: continue
                            sfx = "" if fr is None else f"|{fr:.2f}"
                            out.append((g, f"{cfg}|{task}|{m}{sfx}", dict(cfg=cfg, task=task, method=m, cond=S["cond"], frac=fr)))
    return out


def run_job(g, p, n_eps, gains):
    """回傳含逐 episode 分數的 dict（含 env seed 與控制器設定）。"""
    if g == "EV":
        seed = rq.SEED_EV[p["task"]] + CONF_OFF
        r = rq.run_ev(ALL_CONFIGS[p["cfg"]], p["method"], p["task"], n_eps, gains, seed, p["typ"], p["off"])
        name, y, ckw = rq.method_spec(p["method"], p["off"])
        r.update(env_seed=seed, controller=dict(name=name, yaw_ratio=y["yaw_ratio"], ckw={k: (list(v) if isinstance(v, tuple) else v) for k, v in ckw.items()}))
        return r
    name, nosnap, informed, _ = M[p["method"]]
    cond = p["cond"]
    with seed_shift(CONF_OFF):
        r = p3.run_one(p["cfg"], cond, name, p["task"], n_eps, gains, nosnap=nosnap, informed=informed, frac=p["frac"])
        seed = g1.SEED[p["task"]]
    r.update(env_seed=seed, controller=dict(name=name, nosnap=nosnap, informed=informed, cond=cond, frac=p["frac"]))
    return r


def stage_run(args):
    torch.set_num_threads(4)
    gains = load_gains()
    meta = dict(n_eps=args.n_eps, seed_offset=CONF_OFF, gains=gains.to_dict())
    scr = Path(args.scratch); scr.mkdir(parents=True, exist_ok=True)
    allj = jobs(args)
    parts = {}
    for g in {j[0] for j in allj}:
        f = scr / f"{g}.json"
        d = json.loads(f.read_text()) if f.exists() else {}
        if d.get("_meta") != meta: d = {"_meta": meta, "cells": {}}
        parts[g] = d
    todo = [j for j in allj if j[1] not in parts[j[0]]["cells"]]
    t0, tot = time.time(), len(todo)
    print(f"共 {len(allj)} 格，待跑 {tot}，n_eps={args.n_eps}，seed offset {CONF_OFF}", flush=True)
    for n, (g, k, p) in enumerate(todo):
        t1 = time.time()
        r = run_job(g, p, args.n_eps, gains)
        parts[g]["cells"][k] = r
        (scr / f"{g}.json").write_text(json.dumps(parts[g]))
        dt = time.time() - t1
        print(f"[{n + 1}/{tot}] {g}|{k} score={np.mean(r['score']):.3f} succ={np.mean(r['success']):.2f} "
              f"({dt:.1f}s = {1000 * dt / args.n_eps:.0f} ms/ep，剩約 {(time.time() - t0) / (n + 1) * (tot - n - 1) / 60:.0f} 分)", flush=True)


# ---------------------------------------------------------------------------- 統計與舊值
def ci_mean(x, nb=BOOT_N):
    x = np.asarray(x, float)
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(x), (nb, len(x)))
    lo, hi = np.percentile(x[idx].mean(1), [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def paired_diff(a, b, nb=BOOT_N):
    return ci_mean(np.asarray(a, float) - np.asarray(b, float), nb)


_old = {}


def old_raw(name):
    if name not in _old:
        f = RES / name
        _old[name] = json.loads(f.read_text()) if f.exists() else None
    return _old[name]


def old_score(g, cfg, task, method, p):
    """探索 seed（offset 0）的舊逐 episode 分數與出處；沒有則 (None, 說明)。"""
    if g == "EV":
        raw = old_raw("rqd_classical_raw.json")
        c = (raw or {}).get("cells", {}).get(f"{cfg}|{task}|{p['typ']}-{p['off']}|{method}")
        return (c["score"], "rqd_classical_raw.json") if c else (None, "無舊值")
    tag = M[method][3]
    cond = {"E1": "payload15", "E2": "E2", "E3": "offcenter", "NOM": "nominal"}[g]
    if tag is not None:
        raw = old_raw("p3d_factorial_raw.json")
        k = f"{cond}|{cfg}|{task}|{tag}" + ("" if g != "E2" else f"|{p['frac']:.2f}")
        c = (raw or {}).get("cells", {}).get(k)
        if c: return c["score"], "p3d_factorial_raw.json"
    if g == "E1" and method in ("B1", "B1-QP", "B3 INDI", "B3+adapt"):
        c = (old_raw("g2_raw.json") or {}).get(f"payload15|{cfg}|{method}|{task}")
        if c: return c["score"], "g2_raw.json"
    if g == "E1" and method == "B1-QP-informed":
        c = (old_raw("g2_informed_raw.json") or {}).get(f"payload15|{cfg}|{method}|{task}")
        if c: return c["score"], "g2_informed_raw.json"
    return None, "無舊值"


def sgn(x): return 0 if abs(x) < SIGN_EPS else (1 if x > 0 else -1)


def direction(new, old):
    if old is None or math.isnan(old): return "n/a"
    return "一致" if sgn(new) == sgn(old) else "不一致"


f3 = lambda x: "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"
fs = lambda x: f"{x:+.3f}"


def short(m): return m.replace("(snap=False)", "")


# ---------------------------------------------------------------------------- 報告
def load_parts(scr):
    cells, meta = {}, None
    for g in GROUPS:
        f = Path(scr) / f"{g}.json"
        if not f.exists(): continue
        d = json.loads(f.read_text())
        meta = meta or d["_meta"]
        for k, v in d["cells"].items(): cells[f"{g}|{k}"] = v
    return cells, meta


def get(cells, g, cfg, task, method, frac=None, ev=None):
    if g == "EV": return cells.get(f"EV|{cfg}|{task}|{ev}|{method}")
    sfx = "" if frac is None else f"|{frac:.2f}"
    return cells.get(f"{g}|{cfg}|{task}|{method}{sfx}")


def old_cell(g, cfg, task, method, frac=None, ev=None):
    p = dict(frac=frac)
    if ev: p["typ"], p["off"] = ev.split("-")
    return old_score(g, cfg, task, method, p)


def report_md(cells, meta, out_md):
    L = []
    w = L.append
    n_eps = meta["n_eps"]
    w("# 古典控制器 confirmatory seed 重算（prereg §14(a)）\n")
    w(f"- 來源腳本：`scripts/classical_confirm.py`；每格 {n_eps} episode；confirmatory seed offset {CONF_OFF}（env seed = SEED[task] + {CONF_OFF}；event 用 SEED_EV[task] + {CONF_OFF}）。")
    w("- 控制器參數：g2_estim_tune.json best（estim 一律 snap=False）、g1_coord_tune.json best（yawlim）、rqd_tune.json best（重置偵測）；與探索 seed 相同，未重調。")
    w("- 分數 = success x max(0, 1 - e_rms/e_ref)；區間 = 對 episode 重抽 2000 次的 95% bootstrap 百分位區間（均值）。差的區間為逐 episode 成對差的 bootstrap。")
    w("- 探索 seed 舊值取自 `results/p2/` 下 p3d_factorial_raw.json（含 snap=False 的 `n` 版）、g2_raw.json、g2_informed_raw.json、rqd_classical_raw.json（欄位『出處』）；沒有舊值者標『無舊值』。")
    w(f"- 方向規則（機械）：對比（A-B）的 confirmatory 差與探索差，|差| < {SIGN_EPS} 視為 0；符號相同記『一致』，否則『不一致』，缺舊值記 n/a。")
    w("- RQ-D 的『落在 ±B% 內的比例』：B 在 prereg 未填（P4 未開），此報告不計算；只給每格估計值、CI 與成對差。")
    w("- E2 的 informed 參考不支援（需逐 episode host），E2 只列 B1、B1-QP、B3 系列、estim-yawlim。\n")
    for g, S in GROUPS.items():
        if not any(k.startswith(g + "|") for k in cells): continue
        w(f"\n## {g}：{ENV_DESC[g]}\n")
        evs = [f"{t}-{o}" for t, o in S["conds"]] if g == "EV" else [None]
        fracs = S.get("fracs", (None,))
        hdr = "| 構型 | 任務 | " + ("條件 | " if g == "EV" else "") + ("質量 | " if g == "E2" else "") + "方法 | confirmatory 分數 [95% CI] | 成功率 | 探索 seed 分數 | 出處 |"
        w(hdr); w("|" + "---|" * (hdr.count("|") - 1))
        contrast_rows = []
        for cfg in S["cfgs"]:
            for task in S["tasks"]:
                for ev in evs:
                    for fr in fracs:
                        cur = {}
                        for m in S["methods"]:
                            c = get(cells, g, cfg, task, m, fr, ev)
                            if c is None: continue
                            mu, lo, hi = ci_mean(c["score"])
                            os_, src = old_cell(g, cfg, task, m, fr, ev)
                            om = float(np.mean(os_)) if os_ is not None else None
                            cur[m] = (c, os_)
                            w(f"| {cfg} | {task} | " + (f"{ev} | " if g == "EV" else "") + (f"{fr:.2f} | " if g == "E2" else "")
                              + f"{short(m)} | {mu:.3f} [{lo:.3f}, {hi:.3f}] | {100 * np.mean(c['success']):.0f}% | {f3(om)} | {src} |")
                        for A, B in S["contrasts"]:
                            if A in cur and B in cur:
                                contrast_rows.append((cfg, task, ev, fr, A, B, cur[A], cur[B]))
        if contrast_rows:
            w(f"\n### {g} 成對對比（A - B；逐 episode 成對）\n")
            hdr = "| 構型 | 任務 | " + ("條件 | " if g == "EV" else "") + ("質量 | " if g == "E2" else "") + "A | B | confirmatory 差 [95% CI] | A/B 比 | 探索 seed 差 | 方向 |"
            w(hdr); w("|" + "---|" * (hdr.count("|") - 1))
            for cfg, task, ev, fr, A, B, (ca, oa), (cb, ob) in contrast_rows:
                d, lo, hi = paired_diff(ca["score"], cb["score"])
                ma, mb = float(np.mean(ca["score"])), float(np.mean(cb["score"]))
                ratio = ma / mb if mb > 0.05 else float("nan")
                od = float(np.mean(oa) - np.mean(ob)) if (oa is not None and ob is not None) else None
                w(f"| {cfg} | {task} | " + (f"{ev} | " if g == "EV" else "") + (f"{fr:.2f} | " if g == "E2" else "")
                  + f"{short(A)} | {short(B)} | {fs(d)} [{fs(lo)}, {fs(hi)}] | {f3(ratio)} | {'—' if od is None else fs(od)} | {direction(d, od)} |")
        if g == "EV":
            w("\n### EV 其他指標（均值；峰值 = 事件後 1 s 位置誤差峰值 m；恢復 = 恢復時間中位數 s，未恢復以剩餘時間計）\n")
            w("| 構型 | 任務 | 條件 | 方法 | 峰值 | 恢復(s) | 恢復比例 | 探索 seed 峰值 |"); w("|---|---|---|---|---|---|---|---|")
            oldraw = (old_raw("rqd_classical_raw.json") or {}).get("cells", {})
            for cfg in S["cfgs"]:
                for task in S["tasks"]:
                    for ev in evs:
                        for m in S["methods"]:
                            c = get(cells, g, cfg, task, m, None, ev)
                            if c is None: continue
                            oc = oldraw.get(f"{cfg}|{task}|{ev}|{m}")
                            w(f"| {cfg} | {task} | {ev} | {m} | {np.nanmean(c['peak']):.3f} | {np.median(np.asarray(c['rec_steps']) / 100):.2f} | "
                              f"{np.mean(c['recovered']):.2f} | {f3(float(np.nanmean(oc['peak'])) if oc else None)} |")
            w("\n註：T2 的峰值與恢復時間混入換目標暫態（prereg 5.c 已知限制）。")
        if g == "E3":
            w("\n### E3 T2+POS_STEP 合併（逐 episode 平均 (T2+POS_STEP)/2；次要，不判定）\n")
            w("| 構型 | 方法 | confirmatory [95% CI] | 探索 seed 分數 |"); w("|---|---|---|---|")
            comb = {}
            for cfg in S["cfgs"]:
                for m in S["methods"]:
                    a, b = get(cells, g, cfg, "T2", m), get(cells, g, cfg, "POS_STEP", m)
                    if a is None or b is None: continue
                    x = (np.asarray(a["score"], float) + np.asarray(b["score"], float)) / 2
                    mu, lo, hi = ci_mean(x)
                    oa, _ = old_cell(g, cfg, "T2", m); ob, _ = old_cell(g, cfg, "POS_STEP", m)
                    om = float((np.mean(oa) + np.mean(ob)) / 2) if oa is not None and ob is not None else None
                    comb[(cfg, m)] = (x, om)
                    w(f"| {cfg} | {short(m)} | {mu:.3f} [{lo:.3f}, {hi:.3f}] | {f3(om)} |")
            rows = [(cfg, comb[(cfg, INF_OFF)], comb[(cfg, EST_Y)]) for cfg in S["cfgs"] if (cfg, INF_OFF) in comb and (cfg, EST_Y) in comb]
            if rows:
                w("\n| 構型 | informed-offset - estim-yawlim（T2+POS） [95% CI] | 探索 seed 差 | 方向 |"); w("|---|---|---|---|")
                for cfg, (xa, oa), (xb, ob) in rows:
                    d, lo, hi = paired_diff(xa, xb)
                    od = oa - ob if oa is not None and ob is not None else None
                    w(f"| {cfg} | {fs(d)} [{fs(lo)}, {fs(hi)}] | {'—' if od is None else fs(od)} | {direction(d, od)} |")
            w("\n### E3 一致性檢查（prereg 5.b 已填的 confirmatory 分數，容差 0.002）\n")
            w("| 構型 | 任務 | 方法 | 本次 | prereg 5.b | 差 | 判定 |"); w("|---|---|---|---|---|---|---|")
            for (cfg, task, m), ref in E3_PREREG.items():
                c = get(cells, "E3", cfg, task, m)
                if c is None: continue
                v = float(np.mean(c["score"]))
                w(f"| {cfg} | {task} | {short(m)} | {v:.3f} | {ref:.3f} | {v - ref:+.4f} | {'符合' if abs(v - ref) <= 0.002 else '超出容差'} |")
        if g == "NOM":
            w("\n### 6.1 退化比（payload15 分數 / 標稱分數，同構型同任務同 seed）\n")
            w("| 構型 | 任務 | 方法 | confirmatory payload15 | confirmatory 標稱 | 退化比 | 探索 seed 退化比 |"); w("|---|---|---|---|---|---|---|")
            for cfg in S["cfgs"]:
                for task in S["tasks"]:
                    for m in S["methods"]:
                        n_, p_ = get(cells, "NOM", cfg, task, m), get(cells, "E1", cfg, task, m)
                        if n_ is None or p_ is None: continue
                        mn, mp = float(np.mean(n_["score"])), float(np.mean(p_["score"]))
                        on, _ = old_cell("NOM", cfg, task, m); op, _ = old_cell("E1", cfg, task, m)
                        orat = float(np.mean(op) / np.mean(on)) if on is not None and op is not None and np.mean(on) > 0.05 else None
                        w(f"| {cfg} | {task} | {short(m)} | {mp:.3f} | {mn:.3f} | {f3(mp / mn if mn > 0.05 else float('nan'))} | {f3(orat)} |")
    Path(out_md).write_text("\n".join(L) + "\n")


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def stage_report(args):
    cells, meta = load_parts(args.scratch)
    assert cells, f"{args.scratch} 沒有結果"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    raw = dict(_meta=dict(meta, git_head=git_head(), est=p3.est_params(), yaw=p3.yaw_params(), event_det=rq.det_params(), e3_offset=list(p3.OFFSET),
                          note="confirmatory seed 重算；cells[key] 內含逐 episode score/success/pos_rms/att_rms、env_seed、controller 設定"),
               cells=cells)
    (out / "classical_confirm_raw.json").write_text(json.dumps(raw))
    report_md(cells, meta, out / "classical_confirm.md")
    print("已寫入", out / "classical_confirm_raw.json", out / "classical_confirm.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "report", "all"], default="all")
    ap.add_argument("--only", nargs="+", choices=list(GROUPS), default=None)
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=None, help="只跑這些方法 label（M 的 key，可省略 (snap=False)；EV 用 rqd_classical.METHODS）")
    ap.add_argument("--n-eps", type=int, default=100)
    ap.add_argument("--scratch", default=str(ROOT / ".scratch" / "classical_confirm"))
    ap.add_argument("--out-dir", default=str(RES))
    args = ap.parse_args()
    if args.stage in ("run", "all"): stage_run(args)
    if args.stage in ("report", "all"): stage_report(args)


if __name__ == "__main__":
    main()
