"""G1 關卡：「只看局部形狀的最強古典法」和「知道完整形狀的 oracle」差多少？

構型 × 方法 × 任務，每格 n_eps 個 episode（預設 100），環境 seed 固定（同一構型、任務下所有方法的初始條件相同）。
所有定義在這裡先寫死（看結果之前定的），不依結果調整：

  成功：500 步內沒有終止（env 的 terminated：墜地、位置誤差 > 3 m、傾角 > 100°）。
  誤差：位置誤差 = |goal − pos|；姿態誤差 = 機體姿態 R 與外迴路期望姿態 R_d 的夾角
        （R_d 由共用的外迴路算出，各方法用同一組增益，所以量的是「分配層追不追得上姿態指令」）。
  RMS：對 500 步取均方根。失敗的 episode 終止後的步數用「env 會判定終止的值」補（位置 3 m、姿態 100°），
        所以失敗不會因為提早結束而被低估。
  收斂：ATT_STEP 姿態誤差 < 0.1 rad、POS_STEP 位置誤差 < 0.15 m，且從 t = 2 s 起一直維持到 episode 結束。
        T2 每 1.5 s 換目標，「2 s 內」不適用：改報「每段結束前一步位置誤差 < 0.15 m」的比例（3 個完整段）。
  接頭負載比：每個 episode 取最大值，再對 episode 取中位數與 95 百分位（上限見 module_props.json 的 joint_limits）。
  分數（每個 episode）：success × max(0, 1 − e_rms / e_ref)。
        e = ATT_STEP 用姿態 RMS（e_ref = 1.0 rad）、POS_STEP 與 T2 用位置 RMS（e_ref = 1.5 m）。
        e_ref 約等於「完全不修正」時的初始誤差，所以 1 = 完美，0 = 失敗或完全沒修正。
  正規化分數 = 方法的平均分數 ÷ B1 的平均分數（同構型、同任務）。落後 pp = 100 × (1 − 正規化分數)。

G1 判定（PLAN）：大構型 {line6, line8, grid9, grid10, plus5} × 三種任務共 15 格，
  「最強可行古典法」= 每格取 B2+adapt k=1 與 k=2 中較好的那個；全部 15 格落後都 < 5 pp → RL 沒有空間。
  「較好」是看評估結果挑的，對古典法有利（偏保守）；單獨的 k=1、k=2 也都列出，附成對 bootstrap 95% CI。

每一格的結果存在 results/p2/g1_raw.json（可中斷續跑），最後由它產生 results/p2/g1.md。
執行（專案根目錄）：
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 \
    .venv/bin/python -m scripts.g1_gate [--n-eps 100] [--tag ""] [--configs ..] [--methods ..] [--report-only]"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch

from quadsim.assembly import CONFIGS, TRAIN, LARGE
from quadsim.env import ModularEnv, EnvConfig
from quadsim.params import G
from quadsim.controllers import Gains, make_controller, goal_of, METHOD_NAMES
from scripts.tune_classical import TRAIN_CONFIGS

torch.set_num_threads(1)    # 同 tune_classical：小矩陣在忙碌的機器上單執行緒最穩

RES = Path(__file__).resolve().parent.parent / "results" / "p2"
TASKS = ["ATT_STEP", "POS_STEP", "T2"]
SEED = {t: i for i, t in enumerate(TASKS)}            # 環境 seed（每種任務一個，所有構型、方法共用）
# 事先定好的指標參數（見上方說明）
E_REF = {"ATT_STEP": 1.0, "POS_STEP": 1.5, "T2": 1.5}
CONV_THR = {"ATT_STEP": 0.1, "POS_STEP": 0.15, "T2": 0.15}
CONV_FROM = 200                                        # 2 s
FILL_POS, FILL_ATT = 3.0, math.radians(100)
GAP_PP = 5.0
HELDOUT = ["L3", "T4", "line4", "grid6"]
ROWS = list(TRAIN_CONFIGS) + HELDOUT + LARGE            # TRAIN 的 6 個構型 + 留出 + 大構型
ALL_CONFIGS = {**CONFIGS, **TRAIN_CONFIGS}


# ============================================================================
# 1) 跑一格：(構型, 方法, 任務) → 每個 episode 的指標
# ============================================================================
def run_cell(cells, method, task, n_eps, gains, yaw0=False, twr=None, perturb=None, ckw=None):
    """perturb：物理誤差（env.Perturb 或 dict，G2 用）；ckw：傳給控制器工廠的額外參數。
    yaw0：T2 的 yaw 目標固定為 0（診斷）。twr：把每顆馬達的推力上限改成「整架推重比 = twr」對應的值（診斷）。"""
    env = ModularEnv(cells, n_eps, task=task, seed=SEED[task], cfg=EnvConfig(t2_yaw_max=0.0) if yaw0 else None, perturb=perturb)
    if twr is not None:                                               # TWR = 4 Tmax / (m g)
        env.tmax = twr * env.p["mass"] * G / 4
        env.geo["kmax"] = torch.full_like(env.geo["kmax"], env.tmax)
    ctrl = make_controller(method, gains, **(ckw or {}))
    env.reset()
    st, T = env.state(), env.cfg.episode_steps
    pos_err, att_err, jload = (np.zeros((T, n_eps)) for _ in range(3))
    term = np.zeros((T, n_eps), bool)
    for s in range(T):
        _, _, te, _, info = env.step(ctrl.act(st, goal_of(st), env))
        fs = info["final_state"]                                      # reset 之前的狀態
        pos_err[s] = (fs["goal_pos"] - fs["pos"]).norm(dim=-1).numpy()
        att_err[s] = ctrl.attitude_error(fs, goal_of(fs)).numpy()
        jload[s] = info["joint_load_ratio"].numpy()
        term[s] = te.numpy()
        st = info["state"]
    # 每個環境只算第一個 episode：終止那一步（含）之前有效
    died = np.maximum.accumulate(term, axis=0)
    valid = ~np.vstack([np.zeros((1, n_eps), bool), died[:-1]])      # 第 s 步有效 = 之前沒終止過
    success = ~died[-1]
    pos_f = np.where(valid, pos_err, FILL_POS)
    att_f = np.where(valid, att_err, FILL_ATT)
    pos_rms, att_rms = np.sqrt((pos_f ** 2).mean(0)), np.sqrt((att_f ** 2).mean(0))
    jmax = np.where(valid, jload, 0).max(0)
    if task == "T2":      # 每段結束前一步（goal 在第 149、299、... 步內換，所以是 148、298、448）
        ends = [i * 150 - 2 for i in (1, 2, 3)]
        conv = np.mean([(pos_f[e] < CONV_THR[task]) & valid[e] for e in ends], 0)
    else:
        e = att_f if task == "ATT_STEP" else pos_f
        conv = ((e[CONV_FROM:] < CONV_THR[task]).all(0) & success).astype(float)
    err = att_rms if task == "ATT_STEP" else pos_rms
    score = success * np.clip(1 - err / E_REF[task], 0, None)
    r = lambda a: np.round(a.astype(float), 5).tolist()
    return dict(success=success.astype(int).tolist(), att_rms=r(att_rms), pos_rms=r(pos_rms), conv=r(conv),
                jmax=r(jmax), score=r(score), n_qp=int(getattr(ctrl, "n_qp", 0)))


def load_gains():
    f = RES / "classical_gains.json"
    assert f.exists(), "先跑 scripts/tune_classical.py 產生 results/p2/classical_gains.json"
    return Gains(**json.loads(f.read_text())["gains"])


def run_all(args, raw_path):
    gains = load_gains()
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}
    meta = dict(gains=gains.to_dict(), n_eps=args.n_eps)
    if raw.get("_meta") != meta:                                       # 增益或 episode 數變了：舊結果不能續用
        raw = {"_meta": meta}
    configs = args.configs or ROWS
    methods = args.methods or METHOD_NAMES
    todo = [(c, m, t) for c in configs for t in TASKS for m in methods if f"{c}|{m}|{t}" not in raw]
    t_start = time.time()
    for i, (c, m, t) in enumerate(todo):
        t0 = time.time()
        raw[f"{c}|{m}|{t}"] = run_cell(ALL_CONFIGS[c], m, t, args.n_eps, gains)
        raw_path.write_text(json.dumps(raw))
        done = i + 1
        eta = (time.time() - t_start) / done * (len(todo) - done) / 60
        print(f"[{done}/{len(todo)}] {c:8s} {m:14s} {t:8s} succ={np.mean(raw[f'{c}|{m}|{t}']['success']):.2f} "
              f"score={np.mean(raw[f'{c}|{m}|{t}']['score']):.3f} ({time.time() - t0:.0f}s, 剩約 {eta:.0f} 分)", flush=True)
    return raw, gains


# ============================================================================
# 2) 彙整與報告
# ============================================================================
def pooled(raw, row, method, task, key):
    """一列（TRAIN 為 6 個構型串接）的某指標，所有 episode 串在一起。"""
    cfgs = list(TRAIN_CONFIGS) if row == "TRAIN" else [row]
    return np.concatenate([np.array(raw[f"{c}|{method}|{task}"][key], float) for c in cfgs])


def gap_ci(raw, row, method, task, rng, ref="B1"):
    """落後 pp = 100(1 − mean(分數)/mean(參考分數))，對 episode 做成對 bootstrap 的 95% CI。回傳 (點估計, 下, 上)。"""
    a, b = pooled(raw, row, method, task, "score"), pooled(raw, row, ref, task, "score")
    if b.mean() < 0.05: return (math.nan,) * 3                          # B1 自己幾乎全失敗，比值沒意義
    idx = rng.integers(0, len(a), (1000, len(a)))
    boot = 100 * (1 - a[idx].mean(1) / np.maximum(b[idx].mean(1), 1e-9))
    return 100 * (1 - a.mean() / b.mean()), *np.percentile(boot, [2.5, 97.5])


def fmt(x, nd=2): return "—" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.{nd}f}"


def table(raw, task, methods):
    lines = ["| 構型 | 方法 | 成功率 | 姿態RMS(rad) | 位置RMS(m) | 收斂率 | 接頭負載比 中位/p95 | 分數 | 正規化分數 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in ["TRAIN"] + ROWS:
        if row == "TRAIN" and not all(f"{c}|{methods[0]}|{task}" in raw for c in TRAIN_CONFIGS): continue
        if row != "TRAIN" and f"{row}|{methods[0]}|{task}" not in raw: continue
        base = pooled(raw, row, "B1", task, "score").mean() if all(
            f"{c}|B1|{task}" in raw for c in (TRAIN_CONFIGS if row == "TRAIN" else [row])) else math.nan
        for m in methods:
            g = lambda k: pooled(raw, row, m, task, k)
            j = g("jmax")
            sc = g("score").mean()
            norm = sc / base if base and base > 0.05 else math.nan
            lines.append(f"| {row if m == methods[0] else ''} | {m} | {100 * g('success').mean():.0f}% | {g('att_rms').mean():.3f} | "
                         f"{g('pos_rms').mean():.3f} | {100 * g('conv').mean():.0f}% | {np.median(j):.3f} / {np.percentile(j, 95):.3f} | "
                         f"{sc:.3f} | {fmt(norm, 3)} |")
    return "\n".join(lines)


def large_summary(raw, methods):
    """大構型：各方法在三種任務上的平均正規化分數、平均成功率、接頭負載 p95。"""
    lines = ["| 方法 | " + " | ".join(f"{c} 正規化分數 / 成功率 / 負載p95" for c in LARGE) + " |", "|---|" + "---|" * len(LARGE)]
    for m in methods:
        cells = []
        for c in LARGE:
            ns, ss, jp = [], [], []
            for t in TASKS:
                if f"{c}|{m}|{t}" not in raw: continue
                b = pooled(raw, c, "B1", t, "score").mean()
                ns.append(pooled(raw, c, m, t, "score").mean() / b if b > 0.05 else math.nan)
                ss.append(pooled(raw, c, m, t, "success").mean()); jp.append(np.percentile(pooled(raw, c, m, t, "jmax"), 95))
            cells.append(f"{fmt(np.nanmean(ns) if ns else math.nan, 3)} / {100 * np.mean(ss):.0f}% / {np.mean(jp):.3f}" if ns else "—")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def verdict(raw):
    rng = np.random.default_rng(0)
    lines = ["| 構型 | 任務 | B1 分數 | k=1 落後pp [95% CI] | k=2 落後pp [95% CI] | 最強古典法落後pp | 成功率落後pp (B1−最強) | 參考：B1-QP 落後pp |",
             "|---|---|---|---|---|---|---|---|"]
    best_all, best_qp, missing = [], [], 0
    for c in LARGE:
        for t in TASKS:
            keys = [f"{c}|{m}|{t}" for m in ["B1", "B1-QP", "B2+adapt k=1", "B2+adapt k=2"]]
            if not all(k in raw for k in keys): missing += 1; continue
            g1, g2 = (gap_ci(raw, c, f"B2+adapt k={k}", t, rng) for k in (1, 2))
            b1 = pooled(raw, c, "B1", t, "score").mean()
            cand = [(g[0], k) for g, k in ((g1, 1), (g2, 2)) if not math.isnan(g[0])]
            best = min(cand) if cand else (math.nan, 0)
            sb = 100 * pooled(raw, c, "B1", t, "success").mean()
            sm = 100 * max(pooled(raw, c, f"B2+adapt k={k}", t, "success").mean() for k in (1, 2))
            best_all.append(best[0])
            gq = gap_ci(raw, c, "B1-QP", t, rng)[0]                          # B1 自己比 B1-QP 差多少（負值 = QP 更好）
            ref2 = "B1-QP" if gq < 0 else "B1"                              # 敏感度：分母改用較強的那個 oracle
            alt = [gap_ci(raw, c, f"B2+adapt k={k}", t, rng, ref2)[0] for k in (1, 2)]
            best_qp.append(min(a for a in alt if not math.isnan(a)) if not all(math.isnan(a) for a in alt) else math.nan)
            lines.append(f"| {c} | {t} | {b1:.3f} | {fmt(g1[0], 1)} [{fmt(g1[1], 1)}, {fmt(g1[2], 1)}] | {fmt(g2[0], 1)} [{fmt(g2[1], 1)}, {fmt(g2[2], 1)}] | "
                         f"{fmt(best[0], 1)} (k={best[1]}) | {sb - sm:.0f} | {fmt(-gq, 1)} |")
    ok = [b for b in best_all if not math.isnan(b)]
    n_ge = sum(b >= GAP_PP for b in ok)
    if missing or not ok:
        head = f"**判定未完成**：還有 {missing} 格沒有結果。"
    elif n_ge == 0:
        head = (f"**G1 判定：RL 沒有空間（在目前的任務強度下）。** 15 格全部落後 < {GAP_PP:.0f} pp"
                f"（最大 {max(ok):.1f} pp）。依 PLAN 要沿任務加重階梯往上走（機動角度 → payload → 推力係數誤差 → 接頭限制）。")
    else:
        head = (f"**G1 判定：RL 有空間。** {len(ok)} 格中有 {n_ge} 格落後 ≥ {GAP_PP:.0f} pp（最大 {max(ok):.1f} pp，"
                f"中位數 {np.median(ok):.1f} pp）。")
    ok2 = [b for b in best_qp if not math.isnan(b)]
    if ok2:
        head += (f"\n\n敏感度：B1 在飽和時不如 B1-QP（plain pinv + clip 會讓差動力矩一起被削掉）。分母改用兩個 oracle 中較強的那個，"
                 f"{len(ok2)} 格中有 {sum(b >= GAP_PP for b in ok2)} 格落後 ≥ {GAP_PP:.0f} pp（最大 {max(ok2):.1f} pp，中位數 {np.median(ok2):.1f} pp）。")
    return head, "\n".join(lines)


def write_report(raw, gains, args, path):
    methods = METHOD_NAMES
    head, vt = verdict(raw)
    g = gains.to_dict()
    md = [f"# G1 關卡：最強可行古典法 vs oracle（{time.strftime('%Y-%m-%d')}）", "",
          f"由 `scripts/g1_gate.py` 產生（原始資料 `g1_raw{args.tag}.json`）。每格 {raw['_meta']['n_eps']} 個 episode，固定 seed，"
          f"同一構型與任務下所有方法的初始條件相同。標稱環境：沒有 payload、沒有對接偏差、沒有推力誤差；接頭限制只量不終止。", "",
          f"外迴路增益（`scripts/tune_classical.py` 在 TRAIN 構型上對 B1 調出，所有方法共用）：kp={g['kp']:.2f}、kd={g['kd']:.2f}、"
          f"kR={g['kR']:.1f}、kw={g['kw']:.1f}、yaw 倍率 {g['yaw_ratio']:.2f}、水平加速度上限 {g['a_xy_max']:.2f} m/s²。", "",
          "## G1 判定", "", head, "", vt, "",
          "「最強古典法」= 每格取 B2+adapt k=1、k=2 中較好者（看評估結果挑的，對古典法有利）。落後 pp = 100 × (1 − 正規化分數)；"
          "負值代表比 B1 好；最後一欄是 B1-QP 相對 B1 的落後 pp（負值 = B1-QP 比 B1 好）。CI 是對 episode 的成對 bootstrap（1000 次）。", "",
          "## 大構型總覽（三種任務平均）", "", "格式：正規化分數 / 成功率 / 接頭負載比 p95。", "", large_summary(raw, methods), "",
          "## 指標定義（結果出來之前定的）", "", "```", (__doc__.split("不依結果調整：")[1]).split("每一格的結果存在")[0].rstrip(), "```", ""]
    for t in TASKS:
        md += [f"## {t}", "", table(raw, t, methods), ""]
    path.write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eps", type=int, default=100)
    ap.add_argument("--tag", default="", help="輸出檔名後綴（小規模試跑用，例如 _smoke）")
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    raw_path = RES / f"g1_raw{args.tag}.json"
    if args.report_only:
        raw, gains = json.loads(raw_path.read_text()), load_gains()
        assert raw["_meta"]["gains"] == gains.to_dict(), "g1_raw 的增益與 classical_gains.json 不一致"
    else:
        raw, gains = run_all(args, raw_path)
    write_report(raw, gains, args, RES / f"g1{args.tag}.md")
    print("已寫入", RES / f"g1{args.tag}.md")


if __name__ == "__main__":
    main()
