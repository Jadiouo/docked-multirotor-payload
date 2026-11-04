"""論文補充影片：兩支並排 3D 對照動畫 + failure.gif 轉 MP4。輸出 results/media/（MP4 + GIF）。

影片 1  gap：E3 偏心 payload（p3d_factorial.OFFSET）、line8、POS_STEP；
        左 estim-yawlim（B1-QP-estim-yawlim，snap=False），右 informed-offset-yawlim（知道真實 payload 的特權參考）。
影片 2  event：中途偏心掛上 payload（rqd_classical 的 attach-offcenter 事件，payload 15%、掛點 random、事件時刻 U(1.0,3.5) s）、
        line6、HOVER（rqd_classical 的事件研究含 HOVER，pos_step=0：hover 中發生事件）；
        左 estim-reset-yawlim（estim-yawlim + CUSUM 突變偵測重置），右 estim-yawlim（無重置）。

環境與控制器一律與論文實驗腳本相同（不重寫動力學／控制器）：
  gap   = scripts.classical_confirm.run_job 的 E3 路徑（g1_gate.SEED[POS_STEP] + 900000；參數 = g2_estim_tune.json best +
          g1_coord_tune.json best；estim 一律 snap=False；informed = 質量 15%、host = argmax(x) 模組、offset = OFFSET）。
  event = scripts.rqd_classical.run_ev 的路徑（env seed = SEED_EV[HOVER] + 900000；偵測超參數 = rqd_tune.json best）。

【事先寫死的 episode 選擇規則（不看畫面）】
  對 confirmatory seed 的 100 個 episode，逐 episode 算兩邊分數的差 gap_i = (左 − 右 的定義見各影片)，
  取 100 個 gap 的平均 gbar，選 argmin_i |gap_i − gbar|；同分取 episode 編號最小。
  影片 1 的 gap = informed − estim-yawlim；影片 2 的 gap = estim-reset − estim-yawlim。
  同一個 episode 兩邊用同一個環境 seed／目標（批次環境同 seed，第 k 個環境槽）。
  每次執行都重算 100 episode 分數並驗證與 results/p2/classical_confirm_raw.json 的逐 episode 分數一致（容差 1e-4）。

指令（專案根目錄；CPU）：
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. \\
      .venv/bin/python scripts/render_videos.py [--only gap event failure] [--resim]
暫存：.scratch/render_videos/（模擬結果快取；--resim 強制重跑）。"""
import argparse, dataclasses, json, pickle, subprocess, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

import scripts.g1_gate as g1
import quadsim.controllers as qc
from scripts.g1_gate import ALL_CONFIGS, E_REF, FILL_POS, load_gains, goal_of
from scripts.p3d_factorial import cond_perturb, est_params, yaw_params, OFFSET, host_of
from scripts import rqd_classical as rq
from scripts.render_failure import quat_to_R

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "media"
SCR = ROOT / ".scratch" / "render_videos"
CONF_OFF, N_EPS = 900000, 100
STEP, FPS = 4, 25                       # 每 4 個控制步（0.04 s）一幀 = 25 fps，即時速度播放
TOL = 1e-4


# ---------------------------------------------------------------------------- 模擬（記錄軌跡）
def sim(env, ctrl, task_for_score):
    env.reset()
    st, T = env.state(), env.cfg.episode_steps
    rec = {k: [] for k in ("pos", "q", "thrust", "goal", "term", "d", "dp")}
    for s in range(T):
        rec["d"].append(env.geo["d"].clone().numpy()); rec["dp"].append(env.geo["dp"].clone().numpy())   # 本步用的幾何（事件在 step 開頭換）
        _, _, te, _, info = env.step(ctrl.act(st, goal_of(st), env))
        fs = info["final_state"]
        for k, v in (("pos", fs["pos"]), ("q", fs["q"]), ("thrust", fs["thrust"]), ("goal", fs["goal_pos"])):
            rec[k].append(v.clone().numpy())
        rec["term"].append(te.numpy().copy())
        st = info["state"]
    rec = {k: np.stack(v) for k, v in rec.items()}
    rec["pos_err"] = np.linalg.norm(rec["goal"] - rec["pos"], axis=-1)
    died = np.maximum.accumulate(rec["term"], axis=0)
    valid = ~np.vstack([np.zeros((1, died.shape[1]), bool), died[:-1]])
    rec["success"] = ~died[-1]
    pos_f = np.where(valid, rec["pos_err"], FILL_POS)
    rec["pos_rms"] = np.sqrt((pos_f ** 2).mean(0))
    rec["score"] = rec["success"] * np.clip(1 - rec["pos_rms"] / E_REF["T2" if task_for_score == "HOVER" else task_for_score], 0, None)
    rec["t_end"] = np.where(died.any(0), died.argmax(0), T - 1)
    return rec, env


def sim_gap(label, gains):
    """classical_confirm E3 路徑：line8、offcenter、POS_STEP、seed = SEED[POS_STEP] + 900000。"""
    cells, task = ALL_CONFIGS["line8"], "POS_STEP"
    y = yaw_params()
    g = dataclasses.replace(gains, yaw_ratio=y["yaw_ratio"])
    if label == "estim-yawlim":
        name, ckw = "B1-QP-estim-yawlim", {**dict(rate_c=y["rate_c"]), **est_params(), "snap": False}
    else:
        name, ckw = "B1-QP-informed-offset-yawlim", {**dict(rate_c=y["rate_c"]), "payload_frac": 0.15, "payload_host": host_of(cells), "payload_offset": OFFSET}
    env = g1.ModularEnv(cells, N_EPS, task=task, seed=g1.SEED[task] + CONF_OFF, perturb=cond_perturb("offcenter", cells))
    ctrl = qc.make_controller_p3d_off(name, g, **ckw)
    rec, env = sim(env, ctrl, task)
    rec["t_ev"] = None
    return rec, env.pitch


def sim_event(method, gains):
    """rqd_classical.run_ev 路徑：line6、HOVER、attach-offcenter、seed = SEED_EV[HOVER] + 900000。"""
    cells, task = ALL_CONFIGS["line6"], "HOVER"
    seed = rq.SEED_EV[task] + CONF_OFF
    env = g1.ModularEnv(cells, N_EPS, task="POS_STEP", seed=seed, cfg=rq.EnvConfig(pos_step=0.0),
                        perturb=rq.Perturb(payload_event=rq.event_of("attach", "offcenter")))
    name, y, ckw = rq.method_spec(method, "offcenter")
    ctrl = qc.make_controller_rqd(name, dataclasses.replace(gains, yaw_ratio=y["yaw_ratio"]), **ckw)
    env.reset()
    t_ev = env.t_event.clone().numpy()
    rec, env = sim(env, ctrl, task)
    rec["t_ev"] = t_ev
    return rec, env.pitch


def select_episode(a, b):
    """gap_i = a_i − b_i；選 |gap_i − mean(gap)| 最小者，同分取最小編號（np.argmin 取第一個）。"""
    gap = np.asarray(a, float) - np.asarray(b, float)
    return int(np.argmin(np.abs(gap - gap.mean()))), gap


def check_vs_raw(scores, key, ref_cells):
    ref = np.asarray(ref_cells[key]["score"], float)
    err = float(np.abs(np.asarray(scores, float) - ref).max())
    print(f"  對照 classical_confirm_raw.json[{key}]：逐 episode 最大差 {err:.2e}（容差 {TOL}）  平均 {ref.mean():.4f}")
    assert err < TOL, f"{key} 與已發表逐 episode 分數不一致：{err}"
    return float(ref.mean())


def build(which, resim):
    SCR.mkdir(parents=True, exist_ok=True)
    f = SCR / f"{which}.pkl"
    if f.exists() and not resim:
        return pickle.loads(f.read_bytes())
    torch.set_num_threads(2)
    gains = load_gains()
    ref = json.loads((ROOT / "results/p2/classical_confirm_raw.json").read_text())["cells"]
    if which == "gap":
        L, R = "estim-yawlim", "informed-offset-yawlim"
        (rl, pl), (rr, pr) = sim_gap(L, gains), sim_gap(R, gains)
        keyL, keyR = "E3|line8|POS_STEP|B1-QP-estim-yawlim(snap=False)", "E3|line8|POS_STEP|B1-QP-informed-offset-yawlim"
        ep, gap = select_episode(rr["score"], rl["score"])           # informed − estim
        mL, mR = check_vs_raw(rl["score"], keyL, ref), check_vs_raw(rr["score"], keyR, ref)
        info = dict(left=L, right=R, left_name="B1-QP-estim-yawlim (snap=False)", right_name="B1-QP-informed-offset-yawlim (knows payload)",
                    gap_def="informed - estim-yawlim", cfg="line8", task="POS_STEP")
    else:
        L, R = "estim-reset-yawlim", "estim-yawlim"
        (rl, pl), (rr, pr) = sim_event(L, gains), sim_event(R, gains)
        keyL, keyR = "EV|line6|HOVER|attach-offcenter|estim-reset-yawlim", "EV|line6|HOVER|attach-offcenter|estim-yawlim"
        ep, gap = select_episode(rl["score"], rr["score"])           # reset − no reset
        mL, mR = check_vs_raw(rl["score"], keyL, ref), check_vs_raw(rr["score"], keyR, ref)
        info = dict(left=L, right=R, left_name="estim-reset-yawlim (CUSUM reset)", right_name="estim-yawlim (no reset)",
                    gap_def="estim-reset - estim-yawlim", cfg="line6", task="HOVER")
    print(f"[{which}] 平均分數 左 {rl['score'].mean():.4f} 右 {rr['score'].mean():.4f}；平均差 {gap.mean():+.4f}；選 episode {ep}，"
          f"差 {gap[ep]:+.4f}，左 {rl['score'][ep]:.4f} 右 {rr['score'][ep]:.4f}")
    pick = lambda r: {k: (v[:, ep] if k in ("pos", "q", "thrust", "goal", "term", "d", "dp", "pos_err") else
                          (v[ep] if k in ("success", "pos_rms", "score", "t_end") else (v[ep] if k == "t_ev" and v is not None else None)))
                      for k, v in r.items()}
    out = dict(info=info, ep=ep, gap_ep=float(gap[ep]), gap_mean=float(gap.mean()), mean_left=float(rl["score"].mean()),
               mean_right=float(rr["score"].mean()), ref_mean_left=mL, ref_mean_right=mR, left=pick(rl), right=pick(rr),
               pitch=pl, all_left=rl["score"].tolist(), all_right=rr["score"].tolist())
    f.write_bytes(pickle.dumps(out))
    return out


# ---------------------------------------------------------------------------- 繪圖
def draw_quad(ax, pos, R, goal, trail, d, dp, thrust, pitch, lim, show_payload, title, tcol):
    ax.cla()
    h = pitch * 0.48
    sq = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]])
    polys = [(pos + (R @ (dm[:, None] + sq.T)).T) for dm in d]
    ax.add_collection3d(Poly3DCollection(polys, facecolor="#4c78a8", edgecolor="k", alpha=0.9, linewidths=0.6))
    th = thrust.reshape(len(d), 4)
    for m, dm in enumerate(d):
        c = pos + R @ dm
        tip = c + R @ np.array([0, 0, th[m].sum() * 0.012])
        ax.plot([c[0], tip[0]], [c[1], tip[1]], [c[2], tip[2]], color="#f58518", lw=1.5)
    if show_payload:
        pp = pos + R @ dp
        ax.scatter(*pp, color="#e45756", s=60, marker="s", depthshade=False, edgecolor="k")
    ax.plot(*trail.T, color="#54a24b", lw=1, alpha=0.7)
    ax.scatter(*goal, color="gold", marker="*", s=200, edgecolor="k", depthshade=False)
    ax.set_xlim(lim[0]); ax.set_ylim(lim[1]); ax.set_zlim(*lim[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-62)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=11, color=tcol)


def render(which, V, fps=FPS):
    info, ep = V["info"], V["ep"]
    Ls, Rs = V["left"], V["right"]
    T = Ls["pos"].shape[0]
    t_ev = int(Ls["t_ev"]) if Ls["t_ev"] is not None else None
    allp = np.concatenate([Ls["pos"], Rs["pos"], Ls["goal"][:1]])
    maxd = max(np.abs(Ls["d"]).max(), np.abs(Rs["d"]).max())
    ctr = (allp.min(0) + allp.max(0)) / 2
    half = max(0.6, (allp.max(0) - allp.min(0)).max() / 2 + 0.25, maxd * 0.9 + 0.1)
    lim = [(ctr[0] - half, ctr[0] + half), (ctr[1] - half, ctr[1] + half), (ctr[2] - half, ctr[2] + half)]
    fig = plt.figure(figsize=(10.8, 7.2), dpi=100)
    gs = GridSpec(2, 2, height_ratios=[3.1, 1.0], left=0.0, right=1.0, bottom=0.0, top=0.80, hspace=0.0, wspace=0.0)
    axs = [fig.add_subplot(gs[0, k], projection="3d") for k in range(2)]
    axe = fig.add_axes([0.08, 0.06, 0.88, 0.17])
    cap = {"gap": f"E3 (15% payload at offset (+0.033, -0.033, -0.022) m), line8, POS_STEP, confirmatory seed, episode {ep} (closest to mean gap)",
           "event": f"Payload attached in flight (off-center, 15%), line6, HOVER, confirmatory seed, episode {ep} (closest to mean gap)"}[which]
    sub = {"gap": f"gap = informed - estim-yawlim: episode {V['gap_ep']:+.3f}, mean over 100 episodes {V['gap_mean']:+.3f}  |  mean score: {V['mean_left']:.3f} vs {V['mean_right']:.3f}",
           "event": f"gap = estim-reset - estim-yawlim: episode {V['gap_ep']:+.3f}, mean over 100 episodes {V['gap_mean']:+.3f}  |  mean score: {V['mean_left']:.3f} vs {V['mean_right']:.3f}"}[which]
    fig.text(0.5, 0.975, cap, ha="center", fontsize=10.5, weight="bold")
    fig.text(0.5, 0.942, sub, ha="center", fontsize=8.5, color="#444")
    fig.text(0.5, 0.911, "red square = payload, gold star = goal, green line = path, orange = motor thrust" +
             (f"   |   attach event at t = {t_ev * 0.01:.2f} s (marked on the error plot)" if t_ev is not None else ""),
             ha="center", fontsize=8, color="#444")
    cols = ["#1f77b4", "#d62728"]
    frames_idx = list(range(0, T, STEP)) + [T - 1]
    t = np.arange(T) * 0.01
    out = []
    for i in frames_idx:
        for ax, S, nm, sc, c in ((axs[0], Ls, info["left_name"], V["all_left"][ep], cols[0]), (axs[1], Rs, info["right_name"], V["all_right"][ep], cols[1])):
            ev_on = t_ev is not None and i >= t_ev
            show = (t_ev is None) or ev_on
            ttl = f"{nm}\nt = {i * 0.01:.2f} s    episode score = {sc:.3f}"
            if t_ev is not None and ev_on and i < t_ev + 60:
                ttl += "\nPAYLOAD ATTACHED"
            draw_quad(ax, S["pos"][i], quat_to_R(S["q"][i]), S["goal"][i], S["pos"][:i + 1], S["d"][i], S["dp"][i], S["thrust"][i],
                      V["pitch"], lim, show, ttl, c)
        axe.cla()
        axe.plot(t, Ls["pos_err"] * 100, color="#cccccc", lw=0.8); axe.plot(t, Rs["pos_err"] * 100, color="#cccccc", lw=0.8)
        axe.plot(t[:i + 1], Ls["pos_err"][:i + 1] * 100, color=cols[0], lw=1.6, label=info["left"])
        axe.plot(t[:i + 1], Rs["pos_err"][:i + 1] * 100, color=cols[1], lw=1.6, label=info["right"])
        if t_ev is not None:
            axe.axvline(t_ev * 0.01, color="k", ls="--", lw=1.2)
            axe.text(t_ev * 0.01 + 0.04, axe.get_ylim()[1] * 0.0 + max(Ls["pos_err"].max(), Rs["pos_err"].max()) * 100 * 0.92, "payload attached", fontsize=8, va="top")
        axe.axvline(i * 0.01, color="gray", lw=0.8)
        axe.set_xlim(0, t[-1]); axe.set_ylim(0, max(Ls["pos_err"].max(), Rs["pos_err"].max()) * 100 * 1.05)
        axe.set_xlabel("time (s)", fontsize=8); axe.set_ylabel("position error (cm)", fontsize=8); axe.tick_params(labelsize=7)
        axe.legend(fontsize=7, loc="upper right", frameon=False)
        fig.canvas.draw()
        out.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------- 輸出
def write_mp4(frames, path, fps=FPS, hold=1.0):
    h, w, _ = frames[0].shape
    assert h % 2 == 0 and w % 2 == 0
    frames = frames + [frames[-1]] * int(hold * fps)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    for fr in frames: p.stdin.write(fr.tobytes())
    p.stdin.close(); assert p.wait() == 0
    return frames


def write_gif(frames, path, fps=12, width=720, colors=128, max_mb=5.0):
    sub = frames[::max(1, round(FPS / fps))]
    h = int(round(width * frames[0].shape[0] / frames[0].shape[1] / 2)) * 2
    ims = [Image.fromarray(f).resize((width, h), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=colors) for f in sub]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=True)
    mb = Path(path).stat().st_size / 1e6
    if mb > max_mb and colors > 32:
        return write_gif(frames, path, fps, int(width * 0.85) // 2 * 2, colors // 2, max_mb)
    return mb


def gif_to_mp4(src, dst):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-movflags", "+faststart", str(dst)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=["gap", "event", "failure"], default=["gap", "event", "failure"])
    ap.add_argument("--resim", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = {"gap": "e3_gap_line8", "event": "attach_event_line6"}
    for w in ("gap", "event"):
        if w not in a.only: continue
        V = build(w, a.resim)
        fr = render(w, V)
        fr = write_mp4(fr, OUT / f"{names[w]}.mp4")
        mb = write_gif(fr, OUT / f"{names[w]}.gif")
        print(f"[{w}] episode {V['ep']} -> {names[w]}.mp4 / .gif ({mb:.2f} MB)")
    if "failure" in a.only:
        gif_to_mp4(OUT / "failure.gif", OUT / "failure.mp4")
        print("failure.gif -> failure.mp4")


if __name__ == "__main__":
    main()
