"""P3d：古典控制器三個手段（QP、yawlim、estim）的 2^3 全因子實驗——缺口分解主圖。

所有條件寫死在這裡（看結果之前定的）：
  底：B1（集中式、知道完整標稱形狀）。三個開關：
    QP     = 有界最小平方分配（B1 → B1-QP）；關 = 偽逆 + clip。
    yawlim = YawLimMixin，rate_c 與 yaw_ratio 取自 results/p2/g1_coord_tune.json 的 best（只有開 yawlim 的組合改 yaw_ratio）。
    estim  = B1QPEstim 的質量／重心線上估計，超參數取自 results/p2/g2_estim_tune.json 的 best（同 g2_estim.best_est）。
             關 QP 的 estim = NoQPAllocMixin（同樣的估計，分配改成每環境偽逆 + clip）。
  8 個組合都跑。另外 estim=1 的 4 個組合以 snap=False 重跑（標「nosnap」列；主分析 snap = tune 選中值 True）。
  特權參考 B1-QP-informed-yawlim：知道 payload 質量與掛點模組（不知道偏移、推力誤差、對接偏差）；nominal 條件下 payload=0，
  與 B1-QP-yawlim 完全相同（不另跑）；E2 無 informed（需要逐 episode 的 host）→ 參考改用 8 組合中平均分數最強者。
條件：
  nominal          無誤差
  payload15 (E1)   g2 的 payload15：15%、掛在 argmax(x) 的模組中心（g2_mismatch.perturb_of）
  E2               質量 {0.10, 0.15, 0.20} × 每個 episode 隨機掛點（RandomPayloadEnv，env seed = SEED[task]），每質量 100 episode，三者串接 = 300
  all15            g2 的 all15：payload 15% + 推力 ±10% + 對接偏差 ±5mm、±2°
  offcenter        payload15，掛點 = argmax(x) 模組 + OFFSET_SCALE·(sx·0.4·pitch, sy·0.4·pitch, −0.03 m)；pitch 讀 cad/out/module_props.json；
                   (sx, sy) 由 default_rng(20261031) 抽一次（OFFSET_SIGNS 寫死）。
                   【2026-10-30 重設】原 OFFSET_SCALE=1 時，payload15 + 偏移的懸停最大馬達推力 / Tmax = 0.92（line8；無偏移 0.85），
                   只剩 8% 裕度，連知道偏移的特權參考在 line8 T2 也只有 0.027 分。事先規則：沿同一方向向量掃描比例 s，
                   取 line8 上懸停（LP：min max T，A T = 懸停 wrench，0 ≤ T）可行的最大比例 s_max（--stage check 重算 = 1.4357），
                   OFFSET_SCALE = 0.5·s_max = 0.718（寫死）。此條件的特權參考 = B1-QP-informed-offset-yawlim（知道質量、掛點模組與偏移）。
構型：line6、line7、line8、grid10（鑑別）+ line3、square4（TRAIN 對照）；任務 T2、POS_STEP、ATT_STEP；每格 100 episode，
  環境 seed = g1_gate.SEED[task]（所有組合同初始條件、成對）；分數、成功定義 = g1_gate.run_cell。
分析：因子效應 = 8 組合分數的標準 2^3 對比（主效應 = 其他兩個開關各水準下的平均差；交互作用 = 對比 /4 形式），單位為分數差，
  對 episode 重抽 2000 次（所有組合共用同一組重抽索引）取 95% CI。缺口 = 100(1 − 方法/參考)（先把三個任務的平均分數相除）。
  缺口分解主圖：以 Shapley（6 種開啟順序的平均邊際增益）把「B1 → 全開」的分數增益分給三個手段，順序無關。
執行：CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \\
        .venv/bin/python -m scripts.p3d_factorial --stage run [--n-eps 100] [--shard i/k] | report"""
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
from scripts.g1_gate import RES, ALL_CONFIGS, TASKS, run_cell, load_gains
from scripts.g2_mismatch import perturb_of
from quadsim.env import Perturb
from quadsim.rl_env import RandomPayloadEnv
import quadsim.controllers as qc

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parent.parent
SCR = ROOT / ".scratch" / "p3d"
RAW = RES / "p3d_factorial_raw.json"
CONFIGS = ["line6", "line7", "line8", "grid10", "line3", "square4"]
DISC = ["line6", "line7", "line8", "grid10"]
TRAINC = ["line3", "square4"]
PITCH = json.loads((ROOT / "cad" / "out" / "module_props.json").read_text())["pitch"]
OFFSET_SIGNS = (1, -1)          # = default_rng(20261031).choice([-1, 1], 2)（見 check_offset_signs），抽一次寫死
OFFSET_SCALE = 0.718            # = 0.5 * s_max，s_max = 1.4357（line8 懸停可行的最大比例；規則見 docstring，--stage check 重算）
OFFSET = tuple(OFFSET_SCALE * v for v in (OFFSET_SIGNS[0] * 0.4 * PITCH, OFFSET_SIGNS[1] * 0.4 * PITCH, -0.03))
E2_FRACS = (0.10, 0.15, 0.20)
REF_MIN = 0.05
FIG_TITLE = {"nominal": "nominal", "payload15": "payload15 (E1)", "E2": "E2 (mass {.10,.15,.20} x random host)", "all15": "all15", "offcenter": "off-centre payload15"}
COND_ORDER = ["nominal", "payload15", "E2", "all15", "offcenter"]
COND_DESC = {"nominal": "nominal", "payload15": "payload15 (E1)", "E2": "E2 (質量{.10,.15,.20}×隨機掛點)", "all15": "all15",
             "offcenter": "偏離中心掛點 payload15"}
FACTORS = ["QP", "yawlim", "estim"]
COMBOS = list(itertools.product((0, 1), repeat=3))          # (qp, yaw, est)


def hover_margin(cfg, scale):
    """line8 等構型：payload15 + scale·基準偏移 下，懸停所需的最小「最大單顆推力 / Tmax」（LP；> 1 = 物理上不可懸停）。
    幾何取自 env._sample_geometry（真實 M、tdir、tq）。"""
    from scipy.optimize import linprog
    from quadsim.env import ModularEnv
    from quadsim.params import G
    base = (OFFSET_SIGNS[0] * 0.4 * PITCH, OFFSET_SIGNS[1] * 0.4 * PITCH, -0.03)
    cells = ALL_CONFIGS[cfg]
    env = ModularEnv(cells, 1, task="T2", seed=0, perturb=Perturb(payload_frac=0.15, payload_host=host_of(cells), payload_offset=tuple(scale * v for v in base)))
    g = env.geo
    td, tq, M = g["tdir"][0].double().numpy(), g["tq"][0].double().numpy(), float(g["M"][0])
    n = td.shape[0]
    r = linprog(np.r_[np.zeros(n), 1], A_ub=np.hstack([np.eye(n), -np.ones((n, 1))]), b_ub=np.zeros(n),
                A_eq=np.hstack([np.vstack([td.T, tq.T]), np.zeros((6, 1))]), b_eq=np.r_[0, 0, M * G, 0, 0, 0], bounds=[(0, None)] * (n + 1))
    return r.x[-1] / env.tmax if r.status == 0 else math.inf


def stage_check(args):
    lo, hi = 0.0, 4.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if hover_margin("line8", mid) <= 1 else (lo, mid)
    print(f"s_max(line8) = {lo:.4f}；0.5 s_max = {lo / 2:.4f}（OFFSET_SCALE = {OFFSET_SCALE}）")
    for cfg in CONFIGS:
        print(cfg, " ".join(f"s={s}:{hover_margin(cfg, s):.3f}" for s in (0, 0.5, OFFSET_SCALE, 1.0)))


def check_offset_signs():
    return tuple(int(x) for x in np.random.default_rng(20261031).choice([-1, 1], 2))


def mname(c):
    qp, yw, es = c
    return "B1" + ("-QP" if qp else "") + ("-estim" if es else "") + ("-yawlim" if yw else "")


def mlabel(c, nosnap=False): return "+".join(n for n, b in zip(FACTORS, c) if b) or "B1"
def ckey(c, nosnap=False): return "".join(map(str, c)) + ("n" if nosnap else "")


def est_params():
    b = json.loads((RES / "g2_estim_tune.json").read_text())["best"]
    return {k: b[k] for k in ("warm_c", "warm_m", "lam_m", "eps_m", "lam_c", "eps_c", "snap", "corr")}


def yaw_params():
    return json.loads((RES / "g1_coord_tune.json").read_text())["best"]


def host_of(cells): return int(np.argmax([c[0] for c in cells]))


def cond_perturb(cond, cells):
    if cond == "nominal": return Perturb()
    if cond in ("payload15", "all15"): return perturb_of(cond, cells)
    if cond == "offcenter": return Perturb(payload_frac=0.15, payload_host=host_of(cells), payload_offset=OFFSET)
    raise ValueError(cond)


def run_one(cfg, cond, name, task, n_eps, gains, nosnap=False, informed=False, frac=None):
    """一格（E2 的 frac 指定質量）。回傳精簡 dict。"""
    cells = ALL_CONFIGS[cfg]
    g, ckw = gains, {}
    if "yawlim" in name:
        y = yaw_params()
        g, ckw = dataclasses.replace(gains, yaw_ratio=y["yaw_ratio"]), dict(rate_c=y["rate_c"])
    if "estim" in name:
        ckw = {**ckw, **est_params(), **(dict(snap=False) if nosnap else {})}
    if informed:
        ckw = {**ckw, "payload_frac": 0.15, "payload_host": host_of(cells)}
        if "offset" in name: ckw["payload_offset"] = OFFSET
    orig_mk, orig_env = g1.make_controller, g1.ModularEnv
    g1.make_controller = qc.make_controller_p3d_off
    pt = None
    if cond == "E2":
        g1.ModularEnv = lambda c, n, **kw: RandomPayloadEnv(c, n, frac, frac_choices=(frac,), **{k: v for k, v in kw.items() if k != "perturb"})
    else:
        pt = cond_perturb(cond, cells)
    try:
        r = run_cell(cells, name, task, n_eps, g, perturb=pt, ckw=ckw)
    finally:
        g1.make_controller, g1.ModularEnv = orig_mk, orig_env
    return {k: r[k] for k in ("score", "success", "pos_rms", "att_rms")}


def jobs():
    """所有 (key, 參數)；順序固定，--shard 以序號分。key = cond|cfg|task|method[|frac]"""
    out = []
    for cond in COND_ORDER:
        for cfg in CONFIGS:
            for task in TASKS:
                fr_list = E2_FRACS if cond == "E2" else (None,)
                for fr in fr_list:
                    sfx = "" if fr is None else f"|{fr:.2f}"
                    for c in COMBOS:
                        out.append((f"{cond}|{cfg}|{task}|{ckey(c)}{sfx}", dict(cfg=cfg, cond=cond, name=mname(c), task=task, frac=fr)))
                        if c[2]:
                            out.append((f"{cond}|{cfg}|{task}|{ckey(c, True)}{sfx}", dict(cfg=cfg, cond=cond, name=mname(c), task=task, frac=fr, nosnap=True)))
                    if cond not in ("nominal", "E2"):
                        out.append((f"{cond}|{cfg}|{task}|informed{sfx}", dict(cfg=cfg, cond=cond, name="B1-QP-informed-offset-yawlim" if cond == "offcenter" else "B1-QP-informed-yawlim", task=task, frac=fr, informed=True)))
    return out


def stage_run(args):
    gains = load_gains()
    meta = dict(gains=gains.to_dict(), n_eps=args.n_eps, est=est_params(), yaw=yaw_params(), offset=list(OFFSET))
    SCR.mkdir(parents=True, exist_ok=True)
    si, sk = (int(x) for x in args.shard.split("/"))
    path = SCR / f"raw_{args.tag}s{si}of{sk}.json"
    raw = json.loads(path.read_text()) if path.exists() else {}
    if raw.get("_meta") != meta: raw = {"_meta": meta}
    allj = jobs()
    if args.conds: allj = [j for j in allj if j[0].split("|")[0] in args.conds]
    if args.configs: allj = [j for j in allj if j[0].split("|")[1] in args.configs]
    if args.tasks: allj = [j for j in allj if j[0].split("|")[2] in args.tasks]
    mine = [j for i, j in enumerate(allj) if i % sk == si and j[0] not in raw]
    t0 = time.time()
    for n, (k, p) in enumerate(mine):
        t1 = time.time()
        raw[k] = run_one(n_eps=args.n_eps, gains=gains, **p)
        path.write_text(json.dumps(raw))
        print(f"[{n + 1}/{len(mine)}] {k} succ={np.mean(raw[k]['success']):.2f} score={np.mean(raw[k]['score']):.3f} "
              f"({time.time() - t1:.0f}s, 剩約 {(time.time() - t0) / (n + 1) * (len(mine) - n - 1) / 60:.0f} 分)", flush=True)


# ============================================================================
# 分析
# ============================================================================
def load_raw(tag):
    raw, meta = {}, None
    for f in sorted(SCR.glob(f"raw_{tag}s*of*.json")):
        d = json.loads(f.read_text())
        m = d.pop("_meta", None)
        assert meta is None or meta == m, f"{f} 的 meta 與其他分片不同"
        meta = m
        raw.update(d)
    return raw, meta


def arr(raw, cond, cfg, task, tag, f="score"):
    """該格該組合的逐 episode 陣列；E2 三個質量串接。"""
    if cond == "E2":
        return np.concatenate([np.array(raw[f"{cond}|{cfg}|{task}|{tag}|{fr:.2f}"][f], float) for fr in E2_FRACS])
    return np.array(raw[f"{cond}|{cfg}|{task}|{tag}"][f], float)


def effects(Y):
    """Y: [8, n]（COMBOS 順序）→ dict 效應名 → [n]（逐 episode 的對比，平均後就是效應）。"""
    X = np.array(COMBOS)
    S = 2 * X - 1
    out = {}
    for r in (1, 2, 3):
        for sub in itertools.combinations(range(3), r):
            sgn = np.prod(S[:, list(sub)], axis=1)
            out[":".join(FACTORS[i] for i in sub)] = (sgn[:, None] * Y).sum(0) / 4
    return out


def boot_ci(Y, rng, nb=2000):
    n = Y.shape[1]
    idx = rng.integers(0, n, (nb, n))
    e = effects(Y)
    ci = {}
    for k, v in e.items():
        # 對比的逐 episode 值 v；重抽平均
        b = v[idx].mean(1)
        ci[k] = (float(v.mean()), *np.percentile(b, [2.5, 97.5]))
    return ci


def shapley(sc):
    """sc: dict combo → 分數；回傳三個手段各自的 Shapley 增益。"""
    out = np.zeros(3)
    perms = list(itertools.permutations(range(3)))
    for p in perms:
        cur = [0, 0, 0]
        for f in p:
            nxt = list(cur); nxt[f] = 1
            out[f] += sc[tuple(nxt)] - sc[tuple(cur)]
            cur = nxt
    return out / len(perms)


def mean_scores(raw, cond, cfg, tag_of, tasks=TASKS):
    return {c: float(np.mean([arr(raw, cond, cfg, t, tag_of(c)).mean() for t in tasks])) for c in COMBOS}


def reference(raw, cond, cfg, task, nosnap):
    """回傳 (參考分數陣列, 說明)。nominal：informed = B1-QP-yawlim；E2：無 informed → 8 組合中該條件×構型×任務最強者。"""
    tag = lambda c: ckey(c, nosnap and c[2] == 1)
    if cond == "nominal": return arr(raw, cond, cfg, task, ckey((1, 1, 0))), "B1-QP-yawlim（nominal 下 informed 與它相同）"
    if cond == "E2":
        best = max(COMBOS, key=lambda c: arr(raw, cond, cfg, task, tag(c)).mean())
        return arr(raw, cond, cfg, task, tag(best)), f"最強組合（{mlabel(best)}）"
    return arr(raw, cond, cfg, task, "informed"), "B1-QP-informed+yawlim"


def f3(x): return f"{x:.3f}"


def analyze(raw, nosnap, rng):
    tag = lambda c: ckey(c, nosnap and c[2] == 1)
    res = {}
    for cond in COND_ORDER:
        for cfg in CONFIGS:
            for task in TASKS:
                Y = np.stack([arr(raw, cond, cfg, task, tag(c)) for c in COMBOS])
                S = np.stack([arr(raw, cond, cfg, task, tag(c), "success") for c in COMBOS])
                ref, rdesc = reference(raw, cond, cfg, task, nosnap)
                res[(cond, cfg, task)] = dict(score=Y.mean(1), succ=S.mean(1), ci=boot_ci(Y, rng), ref=float(ref.mean()), rdesc=rdesc, Y=Y)
    return res


def gap_ref(R, cond, cfg):
    """(cond, cfg) 的缺口參考（三任務平均分數）：預設 = 各任務參考的平均；若 < REF_MIN（特權參考本身幾乎全失敗，例如 offcenter 的 informed
    不知道偏移），改用 8 組合中三任務平均分數最強者，並標明。"""
    ref = float(np.mean([R[(cond, cfg, t)]["ref"] for t in TASKS]))
    desc = R[(cond, cfg, TASKS[0])]["rdesc"]
    if ref < REF_MIN:
        sc = [np.mean([R[(cond, cfg, t)]["score"][i] for t in TASKS]) for i in range(8)]
        j = int(np.argmax(sc)); ref, desc = float(sc[j]), f"[參考分數 < {REF_MIN}，改用最強組合 {mlabel(COMBOS[j])}]"
    return ref, desc


def gap(x, ref): return 100 * (1 - x / ref) if ref > 0.02 else math.nan


def report(args):
    raw, meta = load_raw(args.tag)
    need = [k for k, _ in jobs() if k not in raw]
    assert not need, f"還缺 {len(need)} 格，例如 {need[:3]}"
    rng = np.random.default_rng(0)
    A = {ns: analyze(raw, ns, rng) for ns in (False, True)}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "p3d_factorial_raw.json").write_text(json.dumps(dict(_meta=meta, cells=raw)))
    make_fig(raw, A[False])
    write_md(raw, meta, A)
    print("done")


def make_fig(raw, A):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(COND_ORDER), figsize=(21, 4.8), sharey=True)
    col = {"QP": "#4C78A8", "yawlim": "#F58518", "estim": "#54A24B"}
    for ax, cond in zip(axs, COND_ORDER):
        for i, cfg in enumerate(CONFIGS):
            sc = {c: np.mean([A[(cond, cfg, t)]["score"][COMBOS.index(c)] for t in TASKS]) for c in COMBOS}
            ref, _ = gap_ref(A, cond, cfg)
            if ref < 0.02: ax.text(i, 1, "n/a", ha="center", fontsize=8); continue
            sh = shapley(sc)
            g0 = gap(sc[(0, 0, 0)], ref)
            parts = [100 * s / ref for s in sh]                  # 各手段補掉的缺口（pp）
            resid = gap(sc[(1, 1, 1)], ref)                      # 全開後剩下的缺口
            pos = neg = 0.0
            # 由下而上：剩餘缺口、再疊三個手段（負貢獻往下疊）
            for lab, v, c_ in [("residual (all 3 on)", resid, "#BBBBBB")] + [(f, p, col[f]) for f, p in zip(FACTORS, parts)]:
                if v >= 0: ax.bar(i, v, bottom=pos, color=c_, width=0.7, label=lab); pos += v
                else: ax.bar(i, v, bottom=neg, color=c_, width=0.7, label=lab); neg += v
            ax.plot([i - 0.35, i + 0.35], [g0, g0], "k-", lw=1.2)
            ax.text(i, max(pos, g0) + 1, f"{g0:.0f}", ha="center", fontsize=8)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(CONFIGS))); ax.set_xticklabels(CONFIGS, rotation=45, fontsize=8)
        ax.set_title(FIG_TITLE[cond], fontsize=9)
    axs[0].set_ylabel("gap (pp) = 100(1 - method/ref)\nblack line = B1 gap; stack = residual + per-lever reduction")
    h, l = axs[0].get_legend_handles_labels()
    d = dict(zip(l, h)); fig.legend(d.values(), d.keys(), loc="upper right", ncol=4, fontsize=9)
    fig.suptitle("P3d gap decomposition (mean of 3 tasks; Shapley split; ref = B1-QP-informed+yawlim; E2 / failed-ref cells: best of 8 combos)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(RES / "p3d_factorial.png", dpi=130)
    plt.close(fig)


def write_md(raw, meta, A):
    L = [f"# P3d：2^3 因子實驗（QP × yawlim × estim）（{time.strftime('%Y-%m-%d')}）", "",
         "由 `scripts/p3d_factorial.py` 產生；原始資料 `p3d_factorial_raw.json`；圖 `p3d_factorial.png`。條件、構型、任務、參考定義寫死在腳本 docstring。", "",
         f"- yawlim：rate_c={meta['yaw']['rate_c']:.3f}、yaw_ratio={meta['yaw']['yaw_ratio']:.3f}（g1_coord_tune.json best）。",
         f"- estim 超參數（g2_estim_tune.json best）：{ {k: (round(v, 4) if isinstance(v, float) else v) for k, v in meta['est'].items()} }；nosnap 列只把 snap 改 False。",
         f"- offcenter 偏移 = {OFFSET_SCALE}·(±0.4·pitch, ±0.4·pitch, −0.03) = ({OFFSET[0]:.4f}, {OFFSET[1]:.4f}, {OFFSET[2]:.4f}) m（pitch={PITCH}；OFFSET_SCALE = 0.5·s_max，規則見 docstring）。每格 {meta['n_eps']} episode（E2 為 3 質量 × {meta['n_eps']}）。",
         "- 表格格式：分數 / 成功率。參考欄為參考方法分數。"]
    for ns in (False, True):
        L += ["", f"# {'主分析（estim 用 tune 選中的 snap=True）' if not ns else 'snap=False 重跑（estim=1 的 4 個組合改 snap=False）'}", ""]
        for cond in COND_ORDER:
            L += [f"## {COND_DESC[cond]}", ""]
            hdr = "| 構型 | 任務 | " + " | ".join(mlabel(c) for c in COMBOS) + " | 參考 |"
            L += [hdr, "|" + "---|" * (len(COMBOS) + 3)]
            for cfg in CONFIGS:
                for t in TASKS:
                    r = A[ns][(cond, cfg, t)]
                    L.append(f"| {cfg} | {t} | " + " | ".join(f"{r['score'][i]:.3f} / {100 * r['succ'][i]:.0f}%" for i in range(8)) + f" | {r['ref']:.3f} |")
            L += ["", "效應（分數差，[95% CI]；主效應 = 其餘兩開關各水準平均的開−關差）：", "",
                  "| 構型 | 任務 | " + " | ".join(A[ns][(cond, CONFIGS[0], TASKS[0])]["ci"].keys()) + " |", "|" + "---|" * 9]
            for cfg in CONFIGS:
                for t in TASKS:
                    ci = A[ns][(cond, cfg, t)]["ci"]
                    L.append(f"| {cfg} | {t} | " + " | ".join(f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for v in ci.values()) + " |")
            L += ["", "缺口 pp（三任務平均分數相對參考）與 Shapley 分解（各手段補掉的 pp）：", "",
                  "| 構型 | 參考 | B1 缺口 | 全開缺口 | QP 補 | yawlim 補 | estim 補 |", "|---|---|---|---|---|---|---|"]
            for cfg in CONFIGS:
                sc = {c: np.mean([A[ns][(cond, cfg, t)]["score"][COMBOS.index(c)] for t in TASKS]) for c in COMBOS}
                ref, rd = gap_ref(A[ns], cond, cfg)
                sh = shapley(sc)
                L.append(f"| {cfg} | {ref:.3f} ({rd}) | {gap(sc[(0, 0, 0)], ref):.1f} | {gap(sc[(1, 1, 1)], ref):.1f} | "
                         + " | ".join(f"{100 * s / ref:+.1f}" if ref > 0.02 else "n/a" for s in sh) + " |")
            L.append("")
    FACTS = RES / "p3d_factorial_facts.txt"
    if FACTS.exists(): L = ["## 事實陳述（人工撰寫，數字出自下方表；不下結論）", "", FACTS.read_text().strip(), ""] + L
    (RES / "p3d_factorial.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "report", "check"])
    ap.add_argument("--n-eps", type=int, default=100)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--conds", nargs="*")
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--tasks", nargs="*")
    args = ap.parse_args()
    assert OFFSET_SIGNS == check_offset_signs(), f"OFFSET_SIGNS 應為 {check_offset_signs()}"
    {"run": stage_run, "report": report, "check": stage_check}[args.stage](args)


if __name__ == "__main__":
    main()
