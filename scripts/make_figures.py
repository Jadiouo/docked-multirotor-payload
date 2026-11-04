"""Portfolio main figure: results/figures/main.png (no simulation; panels (a), (c1), (c2) are rebuilt from saved raw json, panel (b) uses the numbers hard-coded in E3_TABLE below, transcribed from the exploration-seed table in docs/prereg.md section 5.b; the E3 script output behind them was not saved as json).

Story: classical fixes (QP -> yawlim -> estim) close the payload gap; only E3 off-centre POS_STEP and the
privileged "informed" controller still differ; RL did not close it.

Panels
 (a) E1 payload15 T2 score, line6 / line8.  Sources (per-episode 'score' lists, mean over 100 eps):
       B1, B1-QP                -> results/p2/g2_raw.json            keys payload15|<cfg>|<ctrl>|T2
       B1-QP-yawlim, estim-yawlim, informed-yawlim
                                -> results/p2/g2_estim_raw.json      keys eval|<cfg>|T2|<ctrl>
       informed (B1-QP-informed)-> results/p2/g2_informed_raw.json   keys payload15|<cfg>|B1-QP-informed|T2
       dashed line: B1 without payload -> g2_raw.json nominal|<cfg>|B1|T2
     (line7 has no B1 / B1-QP payload15 entry in g2_raw.json, so it is not drawn.)
 (b) E3 (off-centre 15% payload) estim-yawlim vs informed-offset-yawlim, T2 and POS_STEP, line6/7/8.
     Only available as a table in docs/prereg.md section 5.b (E3 exploration-seed table; seed
     offset 0; format "estim / informed"). Hard-coded below.
 (c1) line6 nominal validation seed (seed_offset 100): from-scratch RL recipes vs B1
     -> results/p3/line6_nom_k2_h4, line6_nom_mb_k2_h4, line6_nom_team_k2_h4 /eval_val.json
        cells["line6|<task>"]["RL"/"B1"]["score"]
 (c2) line6 E3, validation seed: residual RL vs estim-yawlim vs informed
     -> results/p3/res_line6_off_k2_h4/eval_val_E3.json cells["line6|<task>"]["RL"]["score"] and ["vs"][...]["score"]
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P2, P3 = ROOT / "results" / "p2", ROOT / "results" / "p3"
OUT = ROOT / "results" / "figures"

# Okabe-Ito colour-blind-safe palette
C = dict(B1="#999999", QP="#E69F00", QPY="#D55E00", EST="#009E73", INF="#0072B2", INFY="#56B4E9",
         RL="#CC79A7", BLK="#000000")


def load(p):
    return json.loads(Path(p).read_text())


def mean_sem(rec):
    s = np.asarray(rec["score"], float)
    return s.mean(), s.std(ddof=1) / np.sqrt(len(s))


def panel_a(ax):
    g2, es, inf = load(P2 / "g2_raw.json"), load(P2 / "g2_estim_raw.json"), load(P2 / "g2_informed_raw.json")
    cfgs = ["line6", "line8"]
    ctrls = [
        ("B1 (nominal model)", C["B1"], lambda c: g2[f"payload15|{c}|B1|T2"]),
        ("B1-QP", C["QP"], lambda c: g2[f"payload15|{c}|B1-QP|T2"]),
        ("B1-QP-yawlim", C["QPY"], lambda c: es[f"eval|{c}|T2|B1-QP-yawlim"]),
        ("estim-yawlim (snap=True)", C["EST"], lambda c: es[f"eval|{c}|T2|B1-QP-estim-yawlim"]),
        ("informed, no yawlim (privileged)", C["INF"], lambda c: inf[f"payload15|{c}|B1-QP-informed|T2"]),
        ("informed-yawlim (privileged)", C["INFY"], lambda c: es[f"eval|{c}|T2|B1-QP-informed-yawlim"]),
    ]
    w = 0.13
    for j, (name, col, get) in enumerate(ctrls):
        xs, ms, ss = [], [], []
        for i, c in enumerate(cfgs):
            m, s = mean_sem(get(c))
            xs.append(i + (j - (len(ctrls) - 1) / 2) * w); ms.append(m); ss.append(s)
        ax.bar(xs, ms, w * 0.92, yerr=ss, color=col, label=name, capsize=2, error_kw=dict(lw=0.8))
        for x, m, e in zip(xs, ms, ss):
            ax.text(x, m + e + 0.008, f"{m:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    for i, c in enumerate(cfgs):
        nom = mean_sem(g2[f"nominal|{c}|B1|T2"])[0]
        ax.hlines(nom, i - 0.46, i + 0.46, colors=C["BLK"], linestyles="--", lw=1)
        ax.text(i - 0.46, nom + 0.006, f"B1 without payload: {nom:.2f}", fontsize=6.5, va="bottom")
    ax.set_xticks(range(len(cfgs))); ax.set_xticklabels(cfgs)
    ax.set_xlim(-0.6, len(cfgs) - 0.1)
    ax.set_ylabel("T2 score (mean of 100 episodes, +-SEM)")
    ax.set_title("(a) E1: 15% payload, T2 tracking.\nestim-yawlim matches the privileged controller that also has yawlim", fontsize=10)
    ax.set_ylim(0, 0.46)
    ax.legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.9)


# docs/prereg.md section 5.b (exploration-seed table): "T2 estim / informed", "POS_STEP estim / informed"
E3_TABLE = {
    "line6": dict(T2=(0.197, 0.204), POS_STEP=(0.552, 0.595)),
    "line7": dict(T2=(0.158, 0.161), POS_STEP=(0.509, 0.565)),
    "line8": dict(T2=(0.131, 0.136), POS_STEP=(0.410, 0.505)),
}


def panel_b(ax):
    labels, est, inf = [], [], []
    for task in ["T2", "POS_STEP"]:
        for c in ["line6", "line7", "line8"]:
            labels.append(f"{c}\n{task}"); est.append(E3_TABLE[c][task][0]); inf.append(E3_TABLE[c][task][1])
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w / 2, est, w, color=C["EST"], label="estim-yawlim")
    ax.bar(x + w / 2, inf, w, color=C["INF"], label="informed-offset-yawlim (privileged)")
    for xi, e, i in zip(x, est, inf):
        ax.text(xi - w / 2, e + 0.01, f"{e:.2f}", ha="center", fontsize=6.5)
        ax.text(xi + w / 2, i + 0.01, f"{i:.2f}", ha="center", fontsize=6.5)
        if i - e > 0.03:
            ax.text(xi, max(e, i) + 0.05, f"-{100 * (1 - e / i):.0f}%", ha="center", fontsize=7.5,
                    color=C["QPY"], fontweight="bold")
    ax.axvline(2.5, color="0.7", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("score"); ax.set_ylim(0, 0.75)
    ax.set_title("(b) E3: off-centre payload (exploration seeds, offset 0).\nGap remains in POS_STEP (7-19%); T2 gap <= 0.007 absolute", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")


def panel_c1(ax):
    runs = [("RL baseline", "line6_nom_k2_h4"), ("RL + mean-bound", "line6_nom_mb_k2_h4"),
            ("RL + team sample", "line6_nom_team_k2_h4")]
    tasks = ["POS_STEP", "T2"]
    data = {r: load(P3 / d / "eval_val.json")["cells"] for r, d in runs}
    first = data[runs[0][0]]
    series = [("B1", C["B1"], [first[f"line6|{t}"]["B1"]["score"] for t in tasks])]
    cols = ["#CC79A7", "#882255", "#AA4499"]
    for (r, _), col in zip(runs, cols):
        series.append((r, col, [data[r][f"line6|{t}"]["RL"]["score"] for t in tasks]))
    w = 0.2
    for j, (n, col, vals) in enumerate(series):
        xs = np.arange(len(tasks)) + (j - 1.5) * w
        ax.bar(xs, vals, w * 0.92, color=col, label=n)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks)
    ax.set_ylabel("score"); ax.set_ylim(0, 0.8)
    ax.set_title("(c1) RL from scratch, line6, no payload\n(validation seed): all below B1", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")


def panel_c2(ax):
    d = load(P3 / "res_line6_off_k2_h4" / "eval_val_E3.json")["cells"]
    tasks = ["POS_STEP", "T2"]
    est_k, inf_k = "B1-QP-estim-yawlim（snap=False）", "B1-QP-informed-offset-yawlim（特權）"
    series = [("estim-yawlim", C["EST"], [d[f"line6|{t}"]["vs"][est_k]["score"] for t in tasks]),
              ("residual RL (estim-yawlim + 0.2*delta)", C["RL"], [d[f"line6|{t}"]["RL"]["score"] for t in tasks]),
              ("informed-offset-yawlim (privileged)", C["INF"], [d[f"line6|{t}"]["vs"][inf_k]["score"] for t in tasks])]
    w = 0.26
    for j, (n, col, vals) in enumerate(series):
        xs = np.arange(len(tasks)) + (j - 1) * w
        ax.bar(xs, vals, w * 0.92, color=col, label=n)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks)
    ax.set_ylabel("score"); ax.set_ylim(0, 0.85)
    ax.set_title("(c2) Residual RL on E3, line6 (validation seed):\ndoes not close the gap (exploratory)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
    panel_a(axs[0, 0]); panel_b(axs[0, 1]); panel_c1(axs[1, 0]); panel_c2(axs[1, 1])
    fig.suptitle("Modular-quadrotor payload robustness: classical estimation recovers most of the payload gap;\nRL (from scratch or residual) does not close the remainder",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "main.png", dpi=150)
    print("wrote", OUT / "main.png")


if __name__ == "__main__":
    main()
