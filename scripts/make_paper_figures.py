"""Paper figures (IEEE two-column). Every plotted number is read from raw JSON; nothing is hard-coded.

Outputs in paper/figs/: fig2_classical.{pdf,png}, fig3_rl_probe.{pdf,png}, rqd_aggregate.csv
Seed groups are never mixed on one axis:
  Fig 2 (classical)  = confirmatory (seed_offset 900000)  results/p2/classical_confirm_raw.json, estim snap=False
  Fig 3 (RL probe)   = validation   (seed_offset 100)     results/p3/{res_line6_off_k2_h4,..._s8}/eval_val_{nom,E1,E3}.json,
                                                           results/p3/line6_nom_mb_k2_h4{,_s8}/eval_val.json
Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python -m scripts.make_paper_figures
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P2, P3 = ROOT / "results" / "p2", ROOT / "results" / "p3"
OUT = ROOT / "paper" / "figs"
COL1, COL2 = 3.5, 7.16
CFG3 = ["line6", "line7", "line8"]
# Okabe-Ito
C = dict(QP="#E69F00", QPY="#D55E00", EST="#009E73", INF="#0072B2", RL="#CC79A7", B1="#777777")
EST_Y, INF_Y, INF_OFF = "B1-QP-estim-yawlim(snap=False)", "B1-QP-informed-yawlim", "B1-QP-informed-offset-yawlim"
VAL_ESTIM = "B1-QP-estim-yawlim（snap=False）"
VAL_INFO = "B1-QP-informed-offset-yawlim（特權）"
SEEDS = [("seed 7", "", "o"), ("seed 8", "_s8", "s")]   # (label, dir suffix, marker)


CAP = {}


def load(p):
    return json.loads(Path(p).read_text())


BOOT_N, BOOT_SEED = 2000, 0   # same bootstrap as scripts/classical_confirm.ci_mean


def ms(rec):
    """mean and 95% percentile-bootstrap CI of per-episode scores -> (mean, (mean-lo, hi-mean))."""
    x = np.asarray(rec["score"], float)
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(x), (BOOT_N, len(x)))
    lo, hi = np.percentile(x[idx].mean(1), [2.5, 97.5])
    return x.mean(), (x.mean() - lo, hi - x.mean())


def yerr(vals):
    return np.array([[v[1][0] for v in vals], [v[1][1] for v in vals]])


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "axes.linewidth": 0.6})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 1 (system overview; geometry from quadsim.assembly.assemble)
OFFSET_E3 = (0.0333, -0.0333, -0.0215)   # docs/prereg.md:177 (m)


def fig1():
    from matplotlib.patches import Circle, Rectangle
    from quadsim.assembly import CONFIGS, assemble
    from quadsim.module_params import load_module
    props = load(ROOT / "cad" / "out" / "module_props.json")
    mp, pitch, _ = load_module()                     # 與 env.py:152 相同的模組參數（CAD 質量，不是 CF2X）
    rprop = props["prop_radius_m"]
    fig = plt.figure(figsize=(COL2, 2.9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 5.2], wspace=0.04, hspace=0.55, left=0.0, right=0.995,
                          top=0.97, bottom=0.2)
    axi = fig.add_subplot(gs[:, 0]); axi.axis("off")
    iso = ROOT / "paper" / "figs" / "module_iso_crop.png"   # 由 cad/out/module_iso.svg 以 ImageMagick convert 轉出並裁切
    assert iso.exists(), f"缺 {iso}：由 cad/out/module_iso.svg 以 ImageMagick convert 轉 PNG 後裁切"
    axi.imshow(plt.imread(iso)); axi.set_title("(a) CAD module", fontsize=8, pad=2)
    xmax = 0
    axes = []
    for i, name in enumerate(CFG3):
        a = assemble(CONFIGS[name], mp, pitch)
        N, com, pos = a["N"], a["com"], a["rotor_pos"]
        centers = np.array(CONFIGS[name], float) * pitch; centers = (centers - centers.mean(0))   # = module centers rel. CoM
        ax = fig.add_subplot(gs[i, 1]); axes.append(ax)
        for k, c in enumerate(centers):
            ax.add_patch(Rectangle((c[0] - pitch / 2, c[1] - pitch / 2), pitch, pitch, fc="none", ec="#555555", lw=0.6))
            for j in range(4):
                r = pos[4 * k + j]
                ax.add_patch(Circle((r[0], r[1]), rprop, fc="#9ec9e8" if a["rotor_spin"][4 * k + j] > 0 else "#f3c9a0",
                                    ec="#333333", lw=0.4))
        h = int(np.argmax(centers[:, 0])); hc = centers[h]
        e3 = hc[:2] + np.array(OFFSET_E3[:2])
        ax.plot(*hc, "o", ms=5.5, color=C["QPY"], mec="k", mew=0.5, zorder=5)
        ax.plot(*e3, "o", ms=5.5, mfc="white", mec=C["INF"], mew=1.2, zorder=5)
        ax.plot(0, 0, "+", ms=7, color="k", mew=0.8, zorder=5)    # CoM (origin)
        ax.set_aspect("equal"); ax.set_xlim(-0.5 * 8 * pitch - 0.01, 0.5 * 8 * pitch + 0.075); ax.set_ylim(-0.065, 0.085)
        ax.set_yticks([-0.058, 0.058]); ax.set_yticklabels(["-0.058", "0.058"], fontsize=6)
        ax.text(-0.5 * 8 * pitch, 0.066, f"{name}: N={N}, M={a['M']:.3f} kg", fontsize=7, va="bottom", ha="left")
        for sp in ("left", "bottom"): ax.spines[sp].set_visible(False)
        ax.tick_params(length=2)
        if i < 2: ax.set_xticks([])
        ax.set_ylabel("$y$ (m)", fontsize=7, labelpad=0)
    axes[0].set_title("(b) Top view", fontsize=8, pad=10)
    axes[-1].set_xlabel("body $x$ (m), origin = nominal centre of mass (+)", fontsize=7, labelpad=1)
    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].set_xticks(np.arange(-0.4, 0.41, 0.1)); axes[-1].tick_params(labelsize=6)
    h = [plt.Line2D([], [], marker="o", ls="", ms=5.5, color=C["QPY"], mec="k", mew=0.5, label="E1 attach point (host module centre)"),
         plt.Line2D([], [], marker="o", ls="", ms=5.5, mfc="white", mec=C["INF"], mew=1.2,
                    label="E3 attach point (offset $+0.0333,-0.0333$ m; $z$ offset $-0.0215$ m not shown)"),
         plt.Line2D([], [], marker="o", ls="", ms=6, color="#9ec9e8", mec="#333", mew=0.4, label="rotor, spin +1"),
         plt.Line2D([], [], marker="o", ls="", ms=6, color="#f3c9a0", mec="#333", mew=0.4, label="rotor, spin $-$1")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.55, -0.06))
    CAP["fig1"] = (f"Top view (body frame, x forward, y left; z up, E3 z offset not shown) of line6, line7 and line8, drawn from "
                   f"quadsim.assembly.assemble with module pitch {pitch} m and rotor radius {rprop} m "
                   f"(module_props.json). Each module is a square frame with 4 rotors at (+-0.028, +-0.028) m from its centre; "
                   f"x_max module is the payload host. Mass M = N x 0.0381 kg (CAD module mass, module_props.json).\n")
    save(fig, "fig1_overview")


# ---------------------------------------------------------------- Fig 2
def fig2():
    d = load(P2 / "classical_confirm_raw.json"); cells = d["cells"]; meta = d["_meta"]
    n, off = meta["n_eps"], meta["seed_offset"]
    groups = [(c, t) for t in ["T2", "POS_STEP"] for c in CFG3]
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 3.0), gridspec_kw=dict(wspace=0.22))
    ax = axs[0]
    ser = [("B1-QP", C["QP"], "B1-QP"), ("B1-QP-yawlim", C["QPY"], "B1-QP-yawlim"),
           ("estim-yawlim", C["EST"], EST_Y), ("informed-yawlim (priv.)", C["INF"], INF_Y)]
    w = 0.2
    for j, (lab, col, key) in enumerate(ser):
        xs = np.arange(len(groups)) + (j - 1.5) * w
        vals = [ms(cells[f"E1|{c}|{t}|{key}"]) for c, t in groups]
        ax.bar(xs, [v[0] for v in vals], w * 0.92, yerr=yerr(vals), color=col, label=lab, capsize=1.2,
               error_kw=dict(lw=0.6))
    for i, (c, t) in enumerate(groups):   # B1 without payload (NOM, same seed group)
        nom = ms(cells[f"NOM|{c}|{t}|B1-QP-yawlim"])[0]
        ax.hlines(nom, i - 0.42, i + 0.42, colors="k", linestyles="--", lw=0.8,
                  label="B1-QP-yawlim, no payload" if i == 0 else None)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([f"{c[-1]}\n{'T2' if t == 'T2' else 'POS'}" for c, t in groups])
    ax.set_xlabel("line-N configuration (N modules) / task", labelpad=1)
    ax.set_ylabel("score (mean, 95% CI)")
    ax.set_title("(a) E1: 15% centred payload")
    ax.legend(loc="upper left", ncol=1, framealpha=0.9, handlelength=1.2, borderpad=0.3, labelspacing=0.25)
    ax.set_ylim(0, 1.15)

    ax = axs[1]
    ser = [("estim-yawlim", C["EST"], EST_Y), ("informed-offset-yawlim (priv.)", C["INF"], INF_OFF)]
    groups3 = [(c, t) for t in ["T2", "POS_STEP"] for c in CFG3]
    w = 0.36
    gaps = {}
    for j, (lab, col, key) in enumerate(ser):
        xs = np.arange(len(groups3)) + (j - 0.5) * w
        vals = [ms(cells[f"E3|{c}|{t}|{key}"]) for c, t in groups3]
        ax.bar(xs, [v[0] for v in vals], w * 0.92, yerr=yerr(vals), color=col, label=lab, capsize=1.2,
               error_kw=dict(lw=0.6))
        gaps[key] = [v[0] for v in vals]
    for i, (c, t) in enumerate(groups3):
        e, f = gaps[EST_Y][i], gaps[INF_OFF][i]
        g_ = 100 * (f - e) / f
        ax.text(i, max(e, f) + 0.07, (f"$-${g_:.0f}%" if g_ >= 0.5 else "$\\approx$0%"), ha="center", fontsize=8,
                color=C["QPY"])
    ax.set_xticks(range(len(groups3))); ax.set_xticklabels([f"{c[-1]}\n{'T2' if t == 'T2' else 'POS'}" for c, t in groups3])
    ax.set_xlabel("line-N configuration (N modules) / task", labelpad=1)
    ax.set_ylabel("score (mean, 95% CI)")
    ax.set_title("(b) E3: 15% off-centre payload")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", framealpha=0.9, handlelength=1.2, borderpad=0.3, labelspacing=0.25)
    CAP["fig2"] = (f"Confirmatory seeds (seed offset {off}, {n} episodes/cell). Percentages: gap of estim-yawlim "
             "relative to informed-offset-yawlim,\n(inf$-$est)/inf from the plotted means ($\\approx$0%: below 0.5%). "
             "Dashed: B1-QP-yawlim without payload (same seeds). Error bars: 95% bootstrap CI over episodes. POS = POS_STEP.",
             )
    save(fig, "fig2_classical")
    return {f"{c}|{t}": (gaps[EST_Y][i], gaps[INF_OFF][i]) for i, (c, t) in enumerate(groups3)}


# ---------------------------------------------------------------- Fig 3
def rl_markers(ax, vals_by_task, fmt):
    """RL result per task as two markers (one per RL training seed) in an extra slot right of the bars;
    one label 'seed7 / seed8' above the pair."""
    for i, vals in enumerate(vals_by_task):
        for (lab, suf, mk), dx in zip(SEEDS, (-0.07, 0.07)):
            ax.scatter(i + 0.38 + dx, vals[lab], marker=mk, s=26, color=C["RL"], edgecolor="k", lw=0.5, zorder=3)
        ax.text(i + 0.38, max(vals.values()) + 0.035, "\n".join(f"s{l[-1]}: " + fmt.format(vals[l]) for l, _, _ in SEEDS), ha="center",
                va="bottom", fontsize=8)


def fig3():
    tasks = ["POS_STEP", "T2", "ATT_STEP"]
    tasks_b = ["POS_STEP", "T2"]
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 3.0), gridspec_kw=dict(wspace=0.22))
    # (a) nominal line6, mean-bound RL vs classical
    ax = axs[0]
    mb = {lab: load(P3 / f"line6_nom_mb_k2_h4{suf}" / "eval_val.json") for lab, suf, _ in SEEDS}
    ref = mb["seed 7"]["cells"]
    for lab in mb:   # classical reference must be identical across seed dirs
        for t in tasks:
            for k in ("B1", "B1-QP"):
                assert abs(mb[lab]["cells"][f"line6|{t}"][k]["score"] - ref[f"line6|{t}"][k]["score"]) < 1e-9
    bars = [("B1", C["B1"], [ref[f"line6|{t}"]["B1"]["score"] for t in tasks]),
            ("B1-QP", C["QP"], [ref[f"line6|{t}"]["B1-QP"]["score"] for t in tasks])]
    w = 0.2
    for j, (lab, col, v) in enumerate(bars):
        ax.bar(np.arange(len(tasks)) + (j - 1) * w, v, w * 0.92, color=col, label=lab)
    rl_markers(ax, [{lab: mb[lab]["cells"][f"line6|{t}"]["RL"]["score"] for lab, _, _ in SEEDS} for t in tasks], "{:.2f}")
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels([t.replace("_", "\\_") if False else t for t in tasks])
    ax.set_xlim(-0.45, 2.65)
    ax.set_ylabel("score (mean of 100 episodes)")
    for lab, suf, mk in SEEDS:
        ax.scatter([], [], marker=mk, s=26, color=C["RL"], edgecolor="k", lw=0.5, label=f"RL, {lab}")
    ax.set_title("(a) Mean-bound RL, line6, no payload")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", ncol=2, framealpha=0.9, handlelength=1.0, columnspacing=0.8, borderpad=0.3,
              labelspacing=0.25)

    # (b) residual RL on E3
    ax = axs[1]
    res = {lab: load(P3 / f"res_line6_off_k2_h4{suf}" / "eval_val_E3.json") for lab, suf, _ in SEEDS}
    ref = res["seed 7"]["cells"]
    for lab in res:
        for t in tasks_b:
            for k in (VAL_ESTIM, VAL_INFO):
                assert abs(res[lab]["cells"][f"line6|{t}"]["vs"][k]["score"] - ref[f"line6|{t}"]["vs"][k]["score"]) < 1e-9
    bars = [("estim-yawlim", C["EST"], [ref[f"line6|{t}"]["vs"][VAL_ESTIM]["score"] for t in tasks_b]),
            ("informed-off. (priv.)", C["INF"], [ref[f"line6|{t}"]["vs"][VAL_INFO]["score"] for t in tasks_b])]
    for j, (lab, col, v) in enumerate(bars):
        ax.bar(np.arange(2) + (j - 1) * w, v, w * 0.92, color=col, label=lab)
    rl_markers(ax, [{lab: res[lab]["cells"][f"line6|{t}"]["RL"]["score"] for lab, _, _ in SEEDS} for t in tasks_b], "{:.3f}")
    ax.set_xticks(range(2)); ax.set_xticklabels(["POS_STEP", "T2"])
    ax.set_xlim(-0.45, 1.65)
    ax.set_ylabel("score (mean of 100 episodes)")
    for lab, suf, mk in SEEDS:
        ax.scatter([], [], marker=mk, s=26, color=C["RL"], edgecolor="k", lw=0.5, label=f"res. RL, {lab}")
    ax.set_title("(b) Residual RL, line6, off-centre")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", ncol=2, framealpha=0.9, handlelength=1.0, columnspacing=0.8, borderpad=0.3,
              labelspacing=0.25)
    sd = {res[l]["seed_offset"] for l in res} | {mb[l]["seed_offset"] for l in mb}
    ne = {res[l]["n_eps"] for l in res} | {mb[l]["n_eps"] for l in mb}
    assert len(sd) == 1 and len(ne) == 1
    CAP["fig3"] = (f"Validation seeds (seed offset {sd.pop()}, {ne.pop()} episodes/cell); exploratory. Circle = RL training "
             "seed 7, square = RL training seed 8; label = seed 7/seed 8.\nBars: classical controllers evaluated on the same validation episodes. "
             "Error bars: none (the RL/validation JSON files store only per-cell means, no per-episode scores or CI). "
             "Not comparable with Fig. 2 (different seed group).")
    save(fig, "fig3_rl_probe")
    CAP["fig3"] = CAP["fig3"] + " Panel (a) tasks: POS_STEP, T2, ATT_STEP."
    return res, mb


# ---------------------------------------------------------------- RQ-D aggregate
def rqd_csv():
    cells = load(P2 / "classical_confirm_raw.json")["cells"]
    cfgs = ["line6", "line7", "line8", "grid10"]
    tasks = ["T2", "POS_STEP", "HOVER"]
    conds = ["attach-center", "attach-offcenter", "drop-center", "drop-offcenter"]
    meths = ["B1-QP-yawlim", "estim-yawlim", "estim-reset-yawlim", "informed-event-yawlim"]
    rows = []
    for t in tasks:
        for cd in conds:
            m = {me: float(np.mean([np.mean(cells[f"EV|{c}|{t}|{cd}|{me}"]["score"]) for c in cfgs])) for me in meths}
            inf = m["informed-event-yawlim"]
            rows.append(dict(task=t, condition=cd, n_configs=len(cfgs), **{f"mean_{me}": m[me] for me in meths},
                             ratio_estim_reset_yawlim=m["estim-reset-yawlim"] / inf,
                             ratio_estim_yawlim=m["estim-yawlim"] / inf,
                             ratio_B1_QP_yawlim=m["B1-QP-yawlim"] / inf))
    with open(OUT / "rqd_aggregate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    fig1()
    g = fig2()
    res, mb = fig3()
    rows = rqd_csv()
    CAP2 = "".join(CAP["fig2"])
    (OUT / "captions.md").write_text("# Caption notes (generated by scripts/make_paper_figures.py)\n\n"
                                     f"## Fig. 1 (system overview)\n\n{CAP['fig1']}\n"
                                     f"## Fig. 2 (classical, confirmatory)\n\n{CAP2}\n\n"
                                     f"## Fig. 3 (RL probe, validation)\n\n{CAP['fig3']}\n")
    print("Fig2 E3 (est, inf, gap%):")
    for k, (e, f) in g.items():
        print(f"  {k}: {e:.4f} {f:.4f} {100 * (f - e) / f:.1f}%")
    for lab, suf, _ in SEEDS:
        print("Fig3", lab, {t: round(res[lab]['cells'][f'line6|{t}']['RL']['score'], 4) for t in ["POS_STEP", "T2"]},
              {t: round(mb[lab]['cells'][f'line6|{t}']['RL']['score'], 4) for t in ["POS_STEP", "T2"]})
    def rng(sel, key):
        v = [r[key] for r in rows if sel(r)]; return f"{min(v):.3f}-{max(v):.3f}"
    cen = lambda r: r["condition"].endswith("-center"); oc = lambda r: r["condition"].endswith("offcenter")
    print("RQ-D estim-reset all:", rng(lambda r: True, "ratio_estim_reset_yawlim"))
    print("RQ-D estim center:", rng(cen, "ratio_estim_yawlim"), " offcenter:", rng(oc, "ratio_estim_yawlim"))
    print("RQ-D B1QPyawlim center:", rng(cen, "ratio_B1_QP_yawlim"), " offcenter:", rng(oc, "ratio_B1_QP_yawlim"))


if __name__ == "__main__":
    main()
