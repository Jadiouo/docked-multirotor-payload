"""P3d_resolve：解開「P3d 偏離中心（t=0 就有、argmax 掛點）estim 差 informed 4.6–11.3 pp」vs「RQ-D 事件條件 reset 達 informed 的 0.98–1.00」。
固定 payload15 + OFFSET（= p3d_factorial.OFFSET），構型 line6/line8/grid10，任務 T2、POS_STEP，env seed = g1_gate.SEED[task]（同 p3d_factorial）。
方法（全帶 yawlim，參數同 g1_coord_tune）：estim（snap=False，g2_estim_tune best）、estim-reset（+rqd_tune.json 的 k,h,chan,refr）、
  informed（B1QPInformedEvent：知道質量／掛點／偏移；t=0 條件下 t_event=0 = 全程知道，等價 B1QPInformedOffsetYawLim，--stage check 驗證）。
條件 2x2：掛點 {argmax, random(RQ-D 的雜湊)} x 起始 {t0 = 一開始就有, attach = 事件 U(1.0,3.5) s}。
t0 條件由 attach 事件 env 改裝：事件前幾何直接設成事件後（payload 在），t_event = 0 且 ev_done = True（不再觸發；供 informed 讀 ev_mode/t_event/ev_host）。
估計誤差：每步（act 之後）記 |M̂−M|/M、|ĉ−c|（c = −mean_i d_i[:2]，m）；t = 0.5、1、2 s（步 50、100、200）與最後一步，取仍存活 episode 的中位數。
缺口 = 100(1 − 方法分數/informed 分數)；分數 = g1_gate 口徑；CI：對 episode 重抽 2000 次（三方法共用索引）。
用法：CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv/bin/python -m scripts.p3d_resolve --stage check|run|report [--n-eps 100]"""
import argparse, dataclasses, json, time
import numpy as np, torch
import scripts.g1_gate as g1
from scripts.g1_gate import RES, ALL_CONFIGS, SEED, E_REF, FILL_POS, load_gains
from scripts.p3d_factorial import OFFSET
from scripts.rqd_classical import est_params, yaw_params, det_params
from quadsim.env import ModularEnv, Perturb
from quadsim.controllers import goal_of, make_controller_rqd

torch.set_num_threads(1)
CFGS, TASKS = ["line6", "line8", "grid10"], ["T2", "POS_STEP"]
HOSTS, STARTS = ["argmax", "random"], ["t0", "attach"]
METHODS = ["estim", "reset", "informed"]
REC = (50, 100, 200, -1)          # 0.5 s、1 s、2 s、結束
FRAC = 0.15
RAW = RES / "p3d_resolve_raw.json"


class T0Env(ModularEnv):
    """attach 事件 env，但 payload 從 t=0 就在。"""
    def _reset_events(self, idx):
        super()._reset_events(idx)
        for k in self.geo: self.geo[k][idx] = self.geo_alt[k][idx]
        self.t_event[idx] = 0
        self.ev_done[idx] = True


def spec(m):
    y, est = yaw_params(), est_params()
    base = dict(rate_c=y["rate_c"])
    if m == "estim": return "B1-QP-estim-yawlim", y, {**base, **est}
    if m == "reset":
        d = {k: v for k, v in det_params().items() if k in ("k", "h", "chan", "refr")}
        return "B1-QP-estim-reset-yawlim", y, {**base, **est, **d}
    return "B1-QP-informed-event-yawlim", y, {**base, **est, "frac": FRAC, "offset": OFFSET}


def run(cfg, task, host, start, m, n, gains):
    ev = dict(type="attach", t_range=(1.0, 3.5), frac=FRAC, host=host, offset=OFFSET)
    cls = T0Env if start == "t0" else ModularEnv
    env = cls(ALL_CONFIGS[cfg], n, task=task, seed=SEED[task], perturb=Perturb(payload_event=ev))
    name, y, ckw = spec(m)
    ctrl = make_controller_rqd(name, dataclasses.replace(gains, yaw_ratio=y["yaw_ratio"]), **ckw)
    env.reset()
    st, T = env.state(), env.cfg.episode_steps
    pos_err = np.zeros((T, n)); term = np.zeros((T, n), bool)
    em, ec = np.full((T, n), np.nan), np.full((T, n), np.nan)
    for s in range(T):
        a = ctrl.act(st, goal_of(st), env)
        if m != "informed":
            M, d = env.geo["M"].double(), env.geo["d"].double()
            ctrue = -d.mean(1)[:, :2]
            em[s] = ((ctrl.M_hat - M).abs() / M).cpu().numpy()
            ec[s] = (ctrl.c_used - ctrue).norm(dim=-1).cpu().numpy()
        _, _, te, _, info = env.step(a)
        fs = info["final_state"]
        pos_err[s] = (fs["goal_pos"] - fs["pos"]).norm(dim=-1).numpy(); term[s] = te.numpy()
        st = info["state"]
    died = np.maximum.accumulate(term, axis=0)
    valid = ~np.vstack([np.zeros((1, n), bool), died[:-1]])
    success = ~died[-1]
    pf = np.where(valid, pos_err, FILL_POS)
    rms = np.sqrt((pf ** 2).mean(0))
    score = success * np.clip(1 - rms / E_REF[task], 0, None)
    out = dict(score=score.round(5).tolist(), success=success.astype(int).tolist())
    if m != "informed":
        med = lambda a, k: float(np.nanmedian(np.where(valid, a, np.nan)[k])) if valid[k].any() else None
        out["err"] = {str(k): [med(em, k), med(ec, k)] for k in REC}
        if hasattr(ctrl, "n_alarm"):
            fa = ctrl.first_alarm_t.cpu().numpy()
            out["alarm_frac"] = float((fa >= 0).mean()); out["alarm_t_med"] = float(np.median(fa[fa >= 0])) if (fa >= 0).any() else None
    return out


def key(c, t, h, s, m): return f"{c}|{t}|{h}|{s}|{m}"


def stage_run(a):
    gains = load_gains()
    raw = json.loads(RAW.read_text()) if RAW.exists() and not a.fresh else {}
    path = RAW if a.n_eps == 100 else RAW.with_name(f"p3d_resolve_raw_n{a.n_eps}.json")
    if path != RAW: raw = json.loads(path.read_text()) if path.exists() else {}
    jobs = [(c, t, h, s, m) for c in CFGS for t in TASKS for h in HOSTS for s in STARTS for m in METHODS]
    for i, j in enumerate(jobs):
        k = key(*j)
        if k in raw: continue
        t0 = time.time(); raw[k] = run(*j, a.n_eps, gains); path.write_text(json.dumps(raw))
        print(f"[{i + 1}/{len(jobs)}] {k} score={np.mean(raw[k]['score']):.3f} succ={np.mean(raw[k]['success']):.2f} ({time.time() - t0:.0f}s)", flush=True)


def stage_check(a):
    """重現 p3d_factorial 的 offcenter|line8|T2：estim(snap=False) 全開（111n）與 informed。"""
    gains = load_gains()
    ref = json.loads((RES / "p3d_factorial_raw.json").read_text())["cells"]
    for m, rk in (("estim", "111n"), ("informed", "informed")):
        r = run("line8", "T2", "argmax", "t0", m, a.n_eps, gains)
        p = ref[f"offcenter|line8|T2|{rk}"]
        n = a.n_eps
        print(m, "新", np.mean(r["score"]), np.mean(r["success"]), "| p3d raw 前", n, "個 episode", np.mean(p["score"][:n]), np.mean(p["success"][:n]),
              "| 逐 episode 最大差", np.abs(np.array(r["score"]) - np.array(p["score"][:n])).max())


def report(a):
    raw = json.loads(RAW.read_text())
    rng = np.random.default_rng(0)
    n = len(raw[key(CFGS[0], TASKS[0], "argmax", "t0", "estim")]["score"])
    idx = rng.integers(0, n, (2000, n))
    L = [f"# P3d_resolve：偏離中心 estim vs reset vs informed（{time.strftime('%Y-%m-%d')}）", "",
         "由 `scripts/p3d_resolve.py` 產生；原始資料 `p3d_resolve_raw.json`。payload15 + OFFSET (+0.0333, −0.0333, −0.0215) m，全帶 yawlim，estim 用 snap=False，reset 參數取自 rqd_tune.json。"
         f"每格 {n} episode，T2 與 POS_STEP，env seed 同 p3d_factorial。缺口 = 100(1 − 方法/informed)，三任務平均分數相除（構型列）或 6 格 episode 串接（合併列）；[95% CI] = episode 重抽 2000 次（三方法共用索引）。", ""]
    S = lambda c, t, h, s, m: np.array(raw[key(c, t, h, s, m)]["score"])
    gaps = {}
    L += ["## 分數（estim / reset / informed）與缺口 pp", "", "| 掛點 | 起始 | 構型 | estim | reset | informed | 缺口 estim | 缺口 reset |", "|---|---|---|---|---|---|---|---|"]
    for h in HOSTS:
        for s in STARTS:
            pooled = {m: [] for m in METHODS}
            for c in CFGS + ["合併"]:
                cs = CFGS if c == "合併" else [c]
                Y = {m: np.mean([S(cc, t, h, s, m) for cc in cs for t in TASKS], axis=0) if c != "合併" else np.mean([S(cc, t, h, s, m) for cc in cs for t in TASKS], axis=0) for m in METHODS}
                mu = {m: Y[m].mean() for m in METHODS}
                cell = {}
                for m in ("estim", "reset"):
                    g = 100 * (1 - mu[m] / mu["informed"])
                    b = 100 * (1 - Y[m][idx].mean(1) / Y["informed"][idx].mean(1))
                    cell[m] = f"{g:+.1f} [{np.percentile(b, 2.5):+.1f}, {np.percentile(b, 97.5):+.1f}]"
                    gaps[(h, s, c, m)] = g
                L.append(f"| {h} | {s} | {c} | {mu['estim']:.3f} | {mu['reset']:.3f} | {mu['informed']:.3f} | {cell['estim']} | {cell['reset']} |")
    L += ["", "## 估計誤差中位數（只列 t0；格式 M̂ 相對誤差 / ĉ 絕對誤差 mm；各構型×任務的 episode 合併後取中位數）", "",
          "| 掛點 | 方法 | 0.5 s | 1 s | 2 s | 結束 | 有警報比例 / 警報步中位數 |", "|---|---|---|---|---|---|---|"]
    for h in HOSTS:
        for m in ("estim", "reset"):
            cols = []
            for k in REC:
                v = [raw[key(c, t, h, "t0", m)]["err"][str(k)] for c in CFGS for t in TASKS]
                cols.append(f"{np.nanmedian([x[0] for x in v if x[0] is not None]):.3f} / {1000 * np.nanmedian([x[1] for x in v if x[1] is not None]):.1f}")
            al = ""
            if m == "reset":
                al = f"{np.mean([raw[key(c, t, h, 't0', m)]['alarm_frac'] for c in CFGS for t in TASKS]):.2f} / " + str(np.nanmedian([raw[key(c, t, h, 't0', m)]['alarm_t_med'] or np.nan for c in CFGS for t in TASKS]))
            L.append(f"| {h} | {m} | " + " | ".join(cols) + f" | {al} |")
    (RES / "p3d_resolve.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["check", "run", "report"])
    ap.add_argument("--n-eps", type=int, default=100)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    {"check": stage_check, "run": stage_run, "report": report}[a.stage](a)


if __name__ == "__main__":
    main()
