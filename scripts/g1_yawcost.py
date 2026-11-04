"""量 yawlim 在 T2 上犧牲多少 yaw 追蹤。照 g1_gate.run_cell 的環境與步進方式，額外記錄對真實目標 yaw 的誤差。
執行：CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
        .venv/bin/python -m scripts.g1_yawcost [--n-eps 100]"""
import argparse, json, math, time
import numpy as np, torch
from scripts.g1_gate import RES, ALL_CONFIGS, load_gains, SEED, E_REF, FILL_POS
from scripts.g1_coord import method_args
from quadsim.env import ModularEnv
from quadsim.controllers import make_controller, goal_of

torch.set_num_threads(2)
CONFIGS = ["line5", "line6", "line7", "line8", "grid10", "grid9"]
METHODS = ["B1", "B1-QP", "B1-QP-yawlim", "B2-QP-coord-v2+adapt-yawlim k=2", "B2-QP+adapt k=2"]
SHORT = {"B2-QP-coord-v2+adapt-yawlim k=2": "coord-v2+adapt-yawlim k=2"}


def yaw_of(q):
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def cell(c, m, n_eps, gains):
    task = "T2"
    g, ck = method_args(m, gains)
    env = ModularEnv(ALL_CONFIGS[c], n_eps, task=task, seed=SEED[task])
    ctrl = make_controller(m, g, **ck)
    env.reset()
    st, T = env.state(), env.cfg.episode_steps
    pe, ye, term = np.zeros((T, n_eps)), np.zeros((T, n_eps)), np.zeros((T, n_eps), bool)
    for s in range(T):
        _, _, te, _, info = env.step(ctrl.act(st, goal_of(st), env))
        fs = info["final_state"]
        pe[s] = (fs["goal_pos"] - fs["pos"]).norm(dim=-1).numpy()
        d = fs["goal_yaw"] - yaw_of(fs["q"])
        ye[s] = ((d + math.pi) % (2 * math.pi) - math.pi).abs().numpy()
        term[s] = te.numpy()
        st = info["state"]
    died = np.maximum.accumulate(term, axis=0)
    valid = ~np.vstack([np.zeros((1, n_eps), bool), died[:-1]])
    succ = ~died[-1]
    pos_rms = np.sqrt((np.where(valid, pe, FILL_POS) ** 2).mean(0))
    score = succ * np.clip(1 - pos_rms / E_REF[task], 0, None)
    nv = valid.sum(0)
    yaw_rms = np.sqrt((ye ** 2 * valid).sum(0) / np.maximum(nv, 1))          # 只算有效步
    frac = ((ye < 0.1) & valid).sum(0) / np.maximum(nv, 1)
    ends = [i * 150 - 2 for i in (1, 2, 3)]
    end_err = np.concatenate([ye[e][valid[e]] for e in ends])
    dpsi = getattr(ctrl, "dpsi", None)
    dt = env.cfg.dt * env.cfg.substeps
    return dict(success=float(succ.mean()), score=float(score.mean()), yaw_rms_all=float(yaw_rms.mean()),
                yaw_rms_succ=float(yaw_rms[succ].mean()) if succ.any() else math.nan,
                end_med=float(np.median(end_err)), end_lt01=float((end_err < 0.1).mean()),
                frac_lt01=float(frac.mean()), dpsi=None if dpsi is None else float(dpsi),
                rate_rad_s=None if dpsi is None else float(dpsi / dt), dt=float(dt),
                goal_yaw_abs_mean=float(st["goal_yaw"].abs().mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-eps", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    gains = load_gains(); raw = {}; t0 = time.time()
    for c in CONFIGS:
        for m in METHODS:
            raw[f"{c}|{m}"] = r = cell(c, m, a.n_eps, gains)
            print(f"{c:7s} {SHORT.get(m, m):28s} succ={r['success']:.2f} score={r['score']:.3f} yawRMS={r['yaw_rms_all']:.3f} "
                  f"end_med={r['end_med']:.3f} <0.1={r['frac_lt01']:.2f} rate={r['rate_rad_s']} ({time.time() - t0:.0f}s)", flush=True)
    if a.out: json.dump(raw, open(a.out, "w"), indent=1); return
    (RES / "g1_yawcost_raw.json").write_text(json.dumps(dict(n_eps=a.n_eps, cells=raw), indent=1))
    L = ["# G1 yawcost：yawlim 在 T2 上的 yaw 追蹤代價", "",
         f"由 `scripts/g1_yawcost.py` 產生；T2、標稱環境、每格 {a.n_eps} episodes、同 seed（同 g1_gate/g1_coord）。rate_c=0.556、yaw_ratio=0.415（g1_coord_tune.json best）。",
         "yaw 誤差 = |wrap(真實目標 yaw − 實際 yaw)|，只算 episode 終止前的有效步。位置分數的定義同 g1_gate。", "",
         "| 構型 | 方法 | 成功率 | 位置分數 | yaw RMS(rad) 全部/僅成功 | 段末 |yaw 誤差| 中位(rad) | 段末 <0.1 rad 比例 | 全時間 <0.1 rad 比例 | yawlim 速率上限 rad/s (dpsi/步) |",
         "|---|---|---|---|---|---|---|---|---|"]
    L[-1] = "|---|---|---|---|---|---|---|---|---|"
    for c in CONFIGS:
        for m in METHODS:
            r = raw[f"{c}|{m}"]
            rate = "—" if r["dpsi"] is None else f"{r['rate_rad_s']:.2f} ({r['dpsi']:.4f})"
            L.append(f"| {c} | {SHORT.get(m, m)} | {100 * r['success']:.0f}% | {r['score']:.3f} | {r['yaw_rms_all']:.3f} / {r['yaw_rms_succ']:.3f} | "
                     f"{r['end_med']:.3f} | {100 * r['end_lt01']:.0f}% | {100 * r['frac_lt01']:.0f}% | {rate} |")
    L += ["", "## 事實陳述", "", "（待填）"]
    (RES / "g1_yawcost.md").write_text("\n".join(L))
    print("done")


if __name__ == "__main__":
    main()
