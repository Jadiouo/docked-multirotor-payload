"""RQ-D 古典端：中途 payload 事件（attach / drop）下的古典控制器（PLAN v7.1 壓力測試）。

事件：env.Perturb.payload_event（見 quadsim/env.py「payload 事件」）。payload 15%、掛點模組 random（雜湊亂數，同 seed 下各方法成對）、
事件時刻 U(1.0, 3.5) s。條件 = {attach, drop} x {中心, 偏離中心}，偏離中心 offset = p3d_factorial.OFFSET = 0.5·s_max 規則得到的
(+0.0333, −0.0333, −0.0215) m。構型 line6、line7、line8、grid10；任務 T2、POS_STEP、HOVER（POS_STEP 的 pos_step = 0：hover 中發生事件）。
方法（全部帶 yawlim，參數同 g1_coord_tune.json）：
  B1-QP-yawlim          不知道 payload
  estim-yawlim          B1QPEstim，snap=False（超參數同 g2_estim_tune.json best），沒有重置
  estim-reset-yawlim    同上 + 殘差 CUSUM 突變偵測 + 重置（B1QPEstimReset；偵測超參數取自 rqd_tune.json）
  informed-event-yawlim 特權上限：知道事件時刻、掛點、質量比例、偏移（B1QPInformedEvent）
指標（每格 100 episode，env seed = g1_gate.SEED[task]，與驗證用 seed 100+SEED 分開）：
  整段分數／成功率：同 g1_gate 定義（success × (1 − 位置 RMS/1.5)；失敗後用 3 m 補）。
  事件後 1 s 位置誤差峰值：max over [t_ev, t_ev+100) 控制步的位置誤差；該 episode 在此窗內終止 = 3 m（env 的終止值）。
  恢復時間：事件前 0.5 s 位置誤差中位數 b；門檻 = 1.5·max(b, 0.02 m)；從 t_ev 起第一個「之後連續 0.5 s 都在門檻內」的步，到 t_ev 的秒數；
            到 episode 結束都沒恢復（或終止）= 未恢復，以 (500 − t_ev) 步計入（右設限）。另報恢復比例。
  注意：T2 每 1.5 s 換目標（第 150、300 步），事件窗可能與換目標重疊，T2 的峰值與恢復時間混入換目標的暫態。
  CI：對 episode 重抽 2000 次的 95% bootstrap（各方法獨立重抽索引，同一組索引用於同格各指標）。
用法（專案根目錄）：
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
     .venv/bin/python -m scripts.rqd_classical --stage tune|run|report [--n-eps 100] [--shard i/k]"""
import argparse
import dataclasses
import itertools
import json
import math
import time
from pathlib import Path
import numpy as np
import torch

import scripts.g1_gate as g1
from scripts.g1_gate import RES, ALL_CONFIGS, SEED, E_REF, FILL_POS, load_gains
from quadsim.env import ModularEnv, EnvConfig, Perturb
from quadsim.controllers import goal_of, make_controller_rqd

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parent.parent
SCR = ROOT / ".scratch" / "rqd"
PITCH = json.loads((ROOT / "cad" / "out" / "module_props.json").read_text())["pitch"]
OFFSET = tuple(0.718 * v for v in (0.4 * PITCH, -0.4 * PITCH, -0.03))       # = p3d_factorial.OFFSET
FRAC, T_RANGE = 0.15, (1.0, 3.5)
CFGS = ["line6", "line7", "line8", "grid10"]
TASKS = ["T2", "POS_STEP", "HOVER"]
CONDS = [("attach", "center"), ("attach", "offcenter"), ("drop", "center"), ("drop", "offcenter")]
METHODS = ["B1-QP-yawlim", "estim-yawlim", "estim-reset-yawlim", "informed-event-yawlim"]
VAL_SEED_OFF = 100
SEED_EV = {**SEED, "HOVER": 3}                      # HOVER 的 env seed（g1 的 SEED 只有三個任務）
WIN, REC_HOLD, REC_MULT, REC_FLOOR, PRE = 100, 50, 1.5, 0.02, 50
DET_DEFAULT = dict(k=1.0, h=12.0, chan="mc", refr=30)
TUNE_PATH = RES / "rqd_tune.json"


def est_params():
    b = json.loads((RES / "g2_estim_tune.json").read_text())["best"]
    return {k: b[k] for k in ("warm_c", "warm_m", "lam_m", "eps_m", "lam_c", "eps_c", "corr")} | dict(snap=False)


def yaw_params(): return json.loads((RES / "g1_coord_tune.json").read_text())["best"]


def det_params():
    return json.loads(TUNE_PATH.read_text())["best"] if TUNE_PATH.exists() else DET_DEFAULT


def event_of(typ, off):
    return dict(type=typ, t_range=T_RANGE, frac=FRAC, host="random", offset=OFFSET if off == "offcenter" else None)


def method_spec(method, off, det=None):
    """(控制器名稱, 增益修改, ckw)"""
    y, est = yaw_params(), est_params()
    base = dict(rate_c=y["rate_c"])
    if method == "B1-QP-yawlim": return "B1-QP-yawlim", y, base
    if method == "estim-yawlim": return "B1-QP-estim-yawlim", y, {**base, **est}
    if method == "estim-reset-yawlim":
        d = {k: v for k, v in (det or det_params()).items() if k in ("k", "h", "chan", "refr")}
        return "B1-QP-estim-reset-yawlim", y, {**base, **est, **d}
    if method == "informed-event-yawlim":
        return "B1-QP-informed-event-yawlim", y, {**base, **est, "frac": FRAC, "offset": OFFSET if off == "offcenter" else None}
    raise ValueError(method)


# ============================================================================
def metrics_from_traces(pos_err, term, task, t_ev):
    """pos_err、term：[T, B]；t_ev：[B]（第一個 episode 的事件步）。回傳 dict（每個 episode 一個值）。"""
    T, B = pos_err.shape
    died = np.maximum.accumulate(term, axis=0)
    valid = ~np.vstack([np.zeros((1, B), bool), died[:-1]])
    success = ~died[-1]
    pos_f = np.where(valid, pos_err, FILL_POS)
    pos_rms = np.sqrt((pos_f ** 2).mean(0))
    score = success * np.clip(1 - pos_rms / E_REF["T2"], 0, None)
    peak, rec, recd, base = (np.zeros(B) for _ in range(4))
    for b in range(B):
        te = int(t_ev[b])
        if te >= T: peak[b], rec[b], recd[b] = np.nan, np.nan, np.nan; continue
        peak[b] = pos_f[te:te + WIN, b].max()
        base[b] = np.median(pos_f[max(0, te - PRE):te, b])
        thr = REC_MULT * max(base[b], REC_FLOOR)
        ok = pos_f[:, b] <= thr
        ok[:te] = False
        if not success[b]: rec[b], recd[b] = T - te, 0; continue
        # 第一個 k >= te，使 ok[k:k+REC_HOLD] 全為 True（窗超出 episode 尾端的不算）
        run = np.convolve(ok.astype(int), np.ones(REC_HOLD, int), mode="valid") == REC_HOLD
        ks = np.nonzero(run[te:])[0]
        if len(ks): rec[b], recd[b] = ks[0], 1
        else: rec[b], recd[b] = T - te, 0
    r = lambda a: np.round(a.astype(float), 5).tolist()
    return dict(success=success.astype(int).tolist(), score=r(score), pos_rms=r(pos_rms), peak=r(peak), rec_steps=r(rec),
                recovered=r(recd), base=r(base), t_ev=np.asarray(t_ev).astype(int).tolist())


def run_ev(cells, method, task, n_eps, gains, seed, typ, off, det=None, extra=None):
    cfg = EnvConfig(pos_step=0.0) if task == "HOVER" else None
    env = ModularEnv(cells, n_eps, task="POS_STEP" if task == "HOVER" else task, seed=seed, cfg=cfg,
                     perturb=Perturb(payload_event=event_of(typ, off)))
    name, y, ckw = method_spec(method, off, det)
    if extra: ckw = {**ckw, **extra}
    ctrl = make_controller_rqd(name, dataclasses.replace(gains, yaw_ratio=y["yaw_ratio"]), **ckw)
    env.reset()
    t_ev = env.t_event.clone().cpu().numpy()
    st, T = env.state(), env.cfg.episode_steps
    pos_err = np.zeros((T, n_eps)); term = np.zeros((T, n_eps), bool)
    for s in range(T):
        _, _, te, _, info = env.step(ctrl.act(st, goal_of(st), env))
        fs = info["final_state"]
        pos_err[s] = (fs["goal_pos"] - fs["pos"]).norm(dim=-1).numpy(); term[s] = te.numpy()
        st = info["state"]
    out = metrics_from_traces(pos_err, term, task, t_ev)
    if hasattr(ctrl, "first_alarm_t"):       # 偵測診斷（第一個 episode，500 步內沒有 autoreset 後的覆寫）：第一次警報相對事件的延遲（步，負 = 事件前誤報）；沒警報 = -999
        fa = ctrl.first_alarm_t.cpu().numpy()
        out["alarm_lag"] = np.where(fa >= 0, fa - t_ev, -999).astype(int).tolist()
        out["n_alarm"] = ctrl.n_alarm.cpu().numpy().astype(int).tolist()
    return out


# ============================================================================
def stage_tune(args):
    """偵測超參數搜尋（≤20 組，含預設）。驗證 seed = 100 + SEED[task]；構型 line6、line8；任務 T2、POS_STEP；
    條件 attach-center、drop-offcenter、attach-offcenter（含 drop 才能懲罰誤報）；每格 args.tune_eps 個 episode。目標 = 各格平均分數的平均。"""
    gains = load_gains()
    rng = np.random.default_rng(20261101)
    trials = [dict(DET_DEFAULT, note="預設")]
    for _ in range(19):
        trials.append(dict(k=float(rng.choice([0.5, 1.0, 2.0, 3.0])), h=float(rng.choice([6, 12, 25, 50, 100])),
                           chan=str(rng.choice(["m", "mc"])), refr=int(rng.choice([15, 30, 60])), note=""))
    cells = [(c, t, cd) for c in ("line6", "line8") for t in ("T2", "POS_STEP") for cd in (("attach", "center"), ("attach", "offcenter"), ("drop", "offcenter"))]
    path = SCR / "tune_trials.json"
    SCR.mkdir(parents=True, exist_ok=True)
    done = json.loads(path.read_text()) if path.exists() else []
    t0 = time.time()
    for i, tr in enumerate(trials):
        if i < len(done): continue
        res = {}
        for c, t, (typ, off) in cells:
            r = run_ev(ALL_CONFIGS[c], "estim-reset-yawlim", t, args.tune_eps, gains, VAL_SEED_OFF + SEED_EV[t], typ, off, det=tr)
            res[f"{c}|{t}|{typ}-{off}"] = dict(score=float(np.mean(r["score"])), peak=float(np.nanmean(r["peak"])), succ=float(np.mean(r["success"])))
        tr = dict(tr, score=float(np.mean([v["score"] for v in res.values()])), cells=res)
        done.append(tr); path.write_text(json.dumps(done))
        print(f"[{i + 1}/{len(trials)}] k={tr['k']} h={tr['h']} chan={tr['chan']} refr={tr['refr']} score={tr['score']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    # 對照：無重置的 estim（同驗證格）
    ref = {}
    for c, t, (typ, off) in cells:
        r = run_ev(ALL_CONFIGS[c], "estim-yawlim", t, args.tune_eps, gains, VAL_SEED_OFF + SEED_EV[t], typ, off)
        ref[f"{c}|{t}|{typ}-{off}"] = dict(score=float(np.mean(r["score"])), peak=float(np.nanmean(r["peak"])), succ=float(np.mean(r["success"])))
    best = max(done, key=lambda d: d["score"])
    TUNE_PATH.write_text(json.dumps(dict(best={k: best[k] for k in ("k", "h", "chan", "refr")}, best_score=best["score"],
                                         no_reset_score=float(np.mean([v["score"] for v in ref.values()])), no_reset_cells=ref,
                                         trials=done, tune_eps=args.tune_eps, val_seed_offset=VAL_SEED_OFF,
                                         note="驗證 seed = 100 + g1_gate.SEED[task]（評估用 SEED[task]，不重疊）；目標 = 各格平均分數的平均"), indent=1))
    print("best", best["k"], best["h"], best["chan"], best["refr"], best["score"])


def jobs():
    return [f"{c}|{t}|{typ}-{off}|{m}" for t in TASKS for (typ, off) in CONDS for c in CFGS for m in METHODS]


def stage_run(args):
    gains = load_gains()
    det = det_params()
    meta = dict(gains=gains.to_dict(), n_eps=args.n_eps, det=det, est=est_params(), yaw=yaw_params(), offset=list(OFFSET), frac=FRAC)
    SCR.mkdir(parents=True, exist_ok=True)
    si, sk = (int(x) for x in args.shard.split("/"))
    path = SCR / f"raw_s{si}of{sk}.json"
    raw = json.loads(path.read_text()) if path.exists() else {}
    if raw.get("_meta") != meta: raw = {"_meta": meta}
    allj = jobs()
    if args.configs: allj = [j for j in allj if j.split("|")[0] in args.configs]
    if args.tasks: allj = [j for j in allj if j.split("|")[1] in args.tasks]
    mine = [j for i, j in enumerate(allj) if i % sk == si and j not in raw]
    t0 = time.time()
    for n, k in enumerate(mine):
        c, t, cd, m = k.split("|"); typ, off = cd.split("-")
        t1 = time.time()
        raw[k] = run_ev(ALL_CONFIGS[c], m, t, args.n_eps, gains, SEED_EV[t], typ, off, det=det)
        path.write_text(json.dumps(raw))
        r = raw[k]
        print(f"[{n + 1}/{len(mine)}] {k} succ={np.mean(r['success']):.2f} score={np.mean(r['score']):.3f} peak={np.nanmean(r['peak']):.3f} "
              f"({time.time() - t1:.0f}s, 剩約 {(time.time() - t0) / (n + 1) * (len(mine) - n - 1) / 60:.0f} 分)", flush=True)


# ============================================================================
def boot(a, stat, idx):
    a = np.asarray(a, float)
    v = np.array([stat(a[i]) for i in idx])
    return float(stat(a)), float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def load_raw():
    raw, meta = {}, None
    for f in sorted(SCR.glob("raw_s*of*.json")):
        d = json.loads(f.read_text()); m = d.pop("_meta", None)
        meta = meta or m
        raw.update(d)
    return raw, meta


def stage_report(args):
    raw, meta = load_raw()
    n = meta["n_eps"]
    idx = np.random.default_rng(0).integers(0, n, (2000, n))
    (RES / "rqd_classical_raw.json").write_text(json.dumps(dict(_meta=meta, cells=raw)))
    f3 = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.3f}"
    cell = {}
    for k, r in raw.items():
        sc = boot(r["score"], np.mean, idx); su = boot(r["success"], np.mean, idx)
        pk = boot(np.nan_to_num(r["peak"], nan=FILL_POS), np.mean, idx)
        rc = boot(np.asarray(r["rec_steps"]) / 100.0, np.median, idx)
        rf = float(np.mean(r["recovered"]))
        cell[k] = dict(score=sc, succ=su, peak=pk, rec=rc, recfrac=rf)
    lines = ["# RQ-D 古典端：中途 payload 事件（attach / drop）", "",
             f"每格 {n} episode；payload {int(FRAC * 100)}%、掛點 random、事件時刻 U(1.0, 3.5) s；偏離中心 offset = ({OFFSET[0]:+.4f}, {OFFSET[1]:+.4f}, {OFFSET[2]:+.4f}) m。"
             f" 偵測超參數 {meta['det']}（rqd_tune.json）。指標定義見 scripts/rqd_classical.py 檔頭；格內為「平均 [95% CI]」，恢復時間為中位數（未恢復以 episode 剩餘時間計，右設限）。",
             "原始資料：results/p2/rqd_classical_raw.json（逐 episode）。", ""]
    for t in TASKS:
        for (typ, off) in CONDS:
            lines += [f"## {t} / {typ} / {'偏離中心' if off == 'offcenter' else '中心'}", "",
                      "| 構型 | 方法 | 分數 | 成功率 | 事件後 1 s 峰值 (m) | 恢復時間 (s) | 恢復比例 |", "|---|---|---|---|---|---|---|"]
            for c in CFGS:
                for m in METHODS:
                    d = cell.get(f"{c}|{t}|{typ}-{off}|{m}")
                    if not d: continue
                    fmt = lambda v, p=3: f"{v[0]:.{p}f} [{v[1]:.{p}f}, {v[2]:.{p}f}]"
                    lines.append(f"| {c} | {m} | {fmt(d['score'])} | {d['succ'][0] * 100:.0f}% [{d['succ'][1] * 100:.0f}, {d['succ'][2] * 100:.0f}] | "
                                 f"{fmt(d['peak'])} | {fmt(d['rec'], 2)} | {d['recfrac'] * 100:.0f}% |")
            lines.append("")
    # 彙總：構型平均（四個構型的平均，各格權重相同）
    lines += ["## 彙總（四構型平均的分數／峰值，按任務 x 條件）", "", "| 任務 | 條件 | " + " | ".join(f"{m} 分數 / 峰值" for m in METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
    for t in TASKS:
        for (typ, off) in CONDS:
            row = []
            for m in METHODS:
                ds = [cell.get(f"{c}|{t}|{typ}-{off}|{m}") for c in CFGS]
                ds = [d for d in ds if d]
                row.append(f"{np.mean([d['score'][0] for d in ds]):.3f} / {np.mean([d['peak'][0] for d in ds]):.3f}" if ds else "—")
            lines.append(f"| {t} | {typ}-{off} | " + " | ".join(row) + " |")
    # 偵測診斷（estim-reset-yawlim；只計成功的 episode；警報延遲 = 第一次警報 − 事件步，單位控制步 = 0.01 s）
    lines += ["", "## 偵測診斷（estim-reset-yawlim，成功的 episode）", ""]
    for typ in ("attach", "drop"):
        lag, miss, fa, tot = [], 0, 0, 0
        for k, r in raw.items():
            if k.endswith("|estim-reset-yawlim") and f"|{typ}-" in k and "alarm_lag" in r:
                for l, ok in zip(r["alarm_lag"], r["success"]):
                    if not ok: continue
                    tot += 1
                    if l == -999: miss += 1
                    elif l < 0: fa += 1
                    else: lag.append(l)
        if tot:
            lines.append(f"- {typ}：{tot} 個 episode，漏報 {miss}，事件前誤報（第一次警報早於事件）{fa}；警報延遲中位數 {np.median(lag):.0f} 步、90 百分位 {np.percentile(lag, 90):.0f} 步。")
    path = RES / "rqd_classical.md"
    keep = ""
    if path.exists():
        old = path.read_text()
        if "## 事實陳述" in old and "（由報告產生後人工補寫" not in old.split("## 事實陳述")[1]:
            keep = old[old.index("## 事實陳述"):]
    lines += [""]
    path.write_text("\n".join(lines) + "\n" + (keep or "## 事實陳述\n\n（由報告產生後人工補寫，見檔尾。）\n"))
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tune", "run", "report"])
    ap.add_argument("--n-eps", type=int, default=100)
    ap.add_argument("--tune-eps", type=int, default=32)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--tasks", nargs="*")
    args = ap.parse_args()
    {"tune": stage_tune, "run": stage_run, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
