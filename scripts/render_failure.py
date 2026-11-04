"""失敗案例 GIF：E3 偏心 payload（line6、POS_STEP）下 B1-QP 墜機 vs B1-QP-estim-yawlim 穩定。
模擬用專案的 ModularEnv（與 rl_eval.compare_e3 同設定：同 seed、同偏移、snap=False），挑一個 B1-QP 失敗而 estim 存活的 episode，
再用 matplotlib 3D 並排動畫（MuJoCo 不參與：評估環境本身是自製 torch 動力學，不是 MuJoCo）。
指令：CUDA_VISIBLE_DEVICES= PYTHONPATH=. \
        .venv/bin/python scripts/render_failure.py [--ep N] [--out results/media/failure.gif]"""
import argparse, dataclasses, math
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import io

import scripts.g1_gate as g1
import quadsim.controllers as qc
from scripts.g1_gate import ALL_CONFIGS, SEED, load_gains, goal_of
from scripts.p3d_factorial import cond_perturb, est_params, yaw_params, OFFSET

CFG, TASK, N_EPS = "line6", "POS_STEP", 100
LABELS = ["B1-QP", "B1-QP-estim-yawlim"]


def simulate(label, gains):
    cells = ALL_CONFIGS[CFG]
    if label == "B1-QP":
        g, ckw, name = gains, {}, "B1-QP"
    else:
        y = yaw_params()
        g = dataclasses.replace(gains, yaw_ratio=y["yaw_ratio"])
        ckw, name = {**dict(rate_c=y["rate_c"]), **est_params(), "snap": False}, "B1-QP-estim-yawlim"
    env = g1.ModularEnv(cells, N_EPS, task=TASK, seed=SEED[TASK], perturb=cond_perturb("offcenter", cells))
    ctrl = qc.make_controller_p3d_off(name, g, **ckw)
    env.reset()
    st, T = env.state(), env.cfg.episode_steps
    rec = dict(pos=[], q=[], thrust=[], goal=[], term=[])
    for s in range(T):
        _, _, te, _, info = env.step(ctrl.act(st, goal_of(st), env))
        fs = info["final_state"]
        for k, v in (("pos", fs["pos"]), ("q", fs["q"]), ("thrust", fs["thrust"]), ("goal", fs["goal_pos"])):
            rec[k].append(v.clone().numpy())
        rec["term"].append(te.numpy().copy())
        st = info["state"]
    rec = {k: np.stack(v) for k, v in rec.items()}          # [T,B,...]
    died = np.maximum.accumulate(rec["term"], axis=0)
    rec["success"] = ~died[-1]
    rec["t_end"] = np.where(died.any(0), died.argmax(0), T - 1)   # 終止那一步（含）
    d, dp, pitch = env.geo["d"][0].numpy(), env.geo["dp"][0].numpy(), env.pitch
    return rec, d, dp, pitch, env


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def tilt_deg(q):
    return math.degrees(math.acos(np.clip(quat_to_R(q)[2, 2], -1, 1)))


def draw(ax, rec, ep, i, d, dp, pitch, title, ok, lim):
    ax.cla()
    pos, R = rec["pos"][i, ep], quat_to_R(rec["q"][i, ep])
    goal = rec["goal"][0, ep]
    h = pitch * 0.48
    sq = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]])
    polys = [(pos + (R @ (dm[:, None] + sq.T)).T) for dm in d]
    ax.add_collection3d(Poly3DCollection(polys, facecolor="#4c78a8", edgecolor="k", alpha=0.9, linewidths=0.6))
    # 推力箭頭
    th = rec["thrust"][i, ep].reshape(len(d), 4)
    for m, dm in enumerate(d):
        c = pos + R @ dm
        tip = c + R @ np.array([0, 0, th[m].sum() * 0.012])
        ax.plot([c[0], tip[0]], [c[1], tip[1]], [c[2], tip[2]], color="#f58518", lw=1.5)
    pp = pos + R @ (np.mean([0], axis=0) + dp)    # payload 位置（相對 CoM 的 dp）
    ax.scatter(*pp, color="#e45756", s=45, marker="s", depthshade=False)
    ax.plot(*rec["pos"][:i + 1, ep].T, color="#54a24b", lw=1, alpha=0.7)
    ax.scatter(*goal, color="gold", marker="*", s=200, edgecolor="k", depthshade=False)
    ax.set_xlim(lim[0]); ax.set_ylim(lim[1]); ax.set_zlim(0, lim[2][1])
    ax.set_box_aspect((lim[0][1]-lim[0][0], lim[1][1]-lim[1][0], lim[2][1]))
    ax.view_init(elev=22, azim=-62)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.tick_params(labelsize=6)
    gx, gy = np.meshgrid(lim[0], lim[1])
    ax.plot_surface(gx, gy, np.zeros_like(gx), color="#dddddd", alpha=0.35)
    col = "#2ca02c" if ok else "#d62728"
    ax.set_title(title, fontsize=11, color="k")
    return col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=None)
    ap.add_argument("--out", default="results/media/failure.gif")
    ap.add_argument("--fps", type=int, default=20)
    a = ap.parse_args()
    torch.set_num_threads(1)
    gains = load_gains()
    sims = {l: simulate(l, gains) for l in LABELS}
    r0, r1 = sims[LABELS[0]][0], sims[LABELS[1]][0]
    print(f"E3 {CFG} {TASK} n={N_EPS}: B1-QP 存活 {r0['success'].sum()}/{N_EPS}，estim-yawlim 存活 {r1['success'].sum()}/{N_EPS}")
    cand = np.where(~r0["success"] & r1["success"])[0]
    print("B1-QP 失敗且 estim 存活的 episode：", cand.tolist())
    ep = a.ep if a.ep is not None else int(cand[0])
    print("選用 episode", ep)
    meta = {}
    for l in LABELS:
        r = sims[l][0]
        te = int(r["t_end"][ep])
        meta[l] = dict(ok=bool(r["success"][ep]), t_end=te, tilt=tilt_deg(r["q"][te, ep]), pos_err=float(np.linalg.norm(r["goal"][te, ep] - r["pos"][te, ep])), z=float(r["pos"][te, ep][2]))
        print(l, meta[l])
    allp = np.concatenate([sims[l][0]["pos"][:, ep] for l in LABELS] + [r0["goal"][:1, ep]])
    ctr = (allp.min(0) + allp.max(0)) / 2; half = max(0.6, (allp.max(0) - allp.min(0)).max() / 2 + 0.25)
    lim = [(ctr[0]-half, ctr[0]+half), (ctr[1]-half, ctr[1]+half), (0, ctr[2]+half)]
    T = r0["pos"].shape[0]; step = max(1, round(100 / a.fps)); frames = []
    fig = plt.figure(figsize=(9, 4.4), dpi=90)
    axs = [fig.add_subplot(1, 2, k + 1, projection="3d") for k in range(2)]
    for i in list(range(0, T, step)) + [T - 1]:
        for ax, l in zip(axs, LABELS):
            rec, d, dp, pitch, _ = sims[l]
            m = meta[l]; j = min(i, m["t_end"]); t = j * 0.01
            crashed = (not m["ok"]) and i >= m["t_end"]
            status = "FAILED: pos err > 3 m (terminated)" if crashed else ("STABLE" if m["ok"] else "failing...")
            draw(ax, rec, ep, j, d, dp, pitch, f"{l}\nt = {t:.2f} s    {status}", m["ok"], lim)
            ax.title.set_color("#d62728" if (not m["ok"]) else "#2ca02c")
        fig.suptitle(f"E3 off-center payload (+0.033, -0.033, -0.022 m), line6, POS_STEP   [red square = payload, star = goal]", fontsize=9)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=0.84, wspace=0)
        buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128))
    frames += [frames[-1]] * (a.fps // 2)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
    import os; print(a.out, os.path.getsize(a.out) / 1e6, "MB", len(frames), "frames")


if __name__ == "__main__":
    main()
