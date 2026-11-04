"""G2-informed：量「不知道 payload」損失的表現。

特權 oracle B1-QP-informed（知道真實 payload 的質量與位置，用含 payload 的 M、CoM、J、轉子位置算 A；
對接偏差、推力係數誤差仍未知）在 payload10、payload15、all15 三個條件下跑，與 g2_raw.json 中的古典法成對比較
（同 seed、同初始條件；本腳本沿用 g2_mismatch 的 perturb_of 與 g1_gate.run_cell，所以幾何與 T2 目標序列相同）。
最強古典法 = 該格（條件×構型×任務）g2_raw 中平均分數最高的古典法（不含 B5）；注意逐格挑最大值對古典法有利（選擇偏差）。
不知道的代價 pp = 100(1 − 最強古典法分數 / informed 分數)；95% CI = 對 episode 做成對 bootstrap（1000 次，
該格最強古典法固定為點估計選出的那個）。組彙總：先對組內各格取平均分數，再算比值；各格獨立重抽 episode。
執行：CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \\
        .venv/bin/python -m scripts.g2_informed [--n-eps 100] [--tag ""] [--report-only]"""
import argparse
import json
import math
import time
import numpy as np

from scripts.g1_gate import RES, ALL_CONFIGS, run_cell, load_gains
from scripts.g2_mismatch import (CONFIGS, TASKS, TRAIN_G, DISC, CTRL, METHODS, COND_DESC, perturb_of, key, table)

CONDS = ["payload10", "payload15", "all15"]
GROUPS = {"TRAIN（line3、square4）": TRAIN_G, "鑑別組（line6、line8、grid10）": DISC, "對照組（grid9）": CTRL,
          "全部 6 構型": CONFIGS}
CLASSICAL = [m for m in METHODS if m != "B5"]
NAME = "B1-QP-informed"


def run_all(args, path):
    gains = load_gains()
    raw = json.loads(path.read_text()) if path.exists() else {}
    meta = dict(gains=gains.to_dict(), n_eps=args.n_eps)
    if raw.get("_meta") != meta: raw = {"_meta": meta}
    todo = [(cd, c, t) for cd in CONDS for c in CONFIGS for t in TASKS if key(cd, c, NAME, t) not in raw]
    t_start = time.time()
    for i, (cd, c, t) in enumerate(todo):
        t0 = time.time()
        pt = perturb_of(cd, ALL_CONFIGS[c])
        ckw = dict(payload_frac=pt.payload_frac, payload_host=pt.payload_host)
        r = raw[key(cd, c, NAME, t)] = run_cell(ALL_CONFIGS[c], NAME, t, args.n_eps, gains, perturb=pt, ckw=ckw)
        path.write_text(json.dumps(raw))
        eta = (time.time() - t_start) / (i + 1) * (len(todo) - i - 1) / 60
        print(f"[{i + 1}/{len(todo)}] {cd:9s} {c:7s} {t:8s} succ={np.mean(r['success']):.2f} score={np.mean(r['score']):.3f} "
              f"({time.time() - t0:.0f}s, 剩約 {eta:.0f} 分)", flush=True)
    return raw


def sc_arr(g2, cd, c, m, t): return np.array(g2[key(cd, c, m, t)]["score"], float)


def best_classical(g2, cd, c, t):
    return max(CLASSICAL, key=lambda m: sc_arr(g2, cd, c, m, t).mean())


def pp(a_mean, b_mean): return 100 * (1 - a_mean / b_mean) if b_mean > 0.05 else math.nan


def boot_pp(cells, rng, nb=1000):
    """cells：[(best_scores, informed_scores)]（每格 episode 對齊）。回傳 (點估計, 下, 上)。"""
    pt = pp(np.mean([b.mean() for b, _ in cells]), np.mean([i.mean() for _, i in cells]))
    n = len(cells[0][0])
    bm, im = np.zeros(nb), np.zeros(nb)
    for b, i in cells:
        idx = rng.integers(0, n, (nb, n))
        bm += b[idx].mean(1) / len(cells); im += i[idx].mean(1) / len(cells)
    boot = 100 * (1 - bm / np.maximum(im, 1e-9))
    return (pt, *np.percentile(boot, [2.5, 97.5]))


def f(x): return "—" if math.isnan(x) else f"{x:.1f}"
def ci(t): return f"{f(t[0])} [{f(t[1])}, {f(t[2])}]"


def report(g2, inf, path, args):
    rng = np.random.default_rng(0)
    L = [f"# G2-informed：不知道 payload 的代價（{time.strftime('%Y-%m-%d')}）", "",
         "由 `scripts/g2_informed.py` 產生（原始資料 `g2_informed_raw.json`；古典法讀自 `g2_raw.json`，未更動）。"
         f"每格 {inf['_meta']['n_eps']} 個 episode，與 g2 成對（同 seed、同初始條件、同 T2 目標序列）。", "",
         "**informed** = B1-QP-informed：B1-QP 的分配，但 M、J、轉子相對真實 CoM 的位置（A）都用含 payload 的真實模型（外迴路 α 限幅仍用標稱值，和所有方法相同）；"
         "對接偏差與推力係數誤差（all15 才有）仍未知，所以 all15 的 informed 不是完全 oracle。"
         "**最強古典法** = 該格 g2_raw 中平均分數最高的古典法（B1、B1-QP、B1-QP+adapt、B2-QP+adapt k=2、B3 INDI、B3+adapt、B4；不含 B5），"
         "逐格挑最大值，對古典法有利（選擇偏差）。", "",
         "**不知道的代價 pp** = 100(1 − 最強古典法分數 / informed 分數)，正 = 不知道比較差；括號為成對 bootstrap 95% CI（episode 重抽 1000 次）。"
         "格式「分數 / 成功率」。", ""]
    summary = {}
    for cd in CONDS:
        L += [f"## {COND_DESC[cd]}（`{cd}`）", ""]
        rows = []
        for c in CONFIGS:
            for t in TASKS:
                i_s = np.array(inf[key(cd, c, NAME, t)]["score"], float)
                i_u = 100 * np.mean(inf[key(cd, c, NAME, t)]["success"])
                bm = best_classical(g2, cd, c, t)
                b_s = sc_arr(g2, cd, c, bm, t)
                b_u = 100 * np.mean(g2[key(cd, c, bm, t)]["success"])
                q = sc_arr(g2, cd, c, "B1-QP", t)
                rows.append([c, t, f"{i_s.mean():.3f} / {i_u:.0f}%", f"{b_s.mean():.3f} / {b_u:.0f}%", bm,
                             f"{q.mean():.3f}", ci(boot_pp([(b_s, i_s)], rng))])
        L += [table(["構型", "任務", "informed", "最強古典法（不知道）", "是哪個", "B1-QP（不知道）", "不知道的代價 pp [95% CI]"], rows), ""]
        grows = []
        for gname, scope in GROUPS.items():
            for t in TASKS + [None]:
                ts = TASKS if t is None else [t]
                cells = [(sc_arr(g2, cd, c, best_classical(g2, cd, c, tt), tt), np.array(inf[key(cd, c, NAME, tt)]["score"], float))
                         for c in scope for tt in ts]
                r = boot_pp(cells, rng)
                summary[(cd, gname, t or "兩任務平均")] = r
                grows.append([gname, t or "兩任務平均", f"{np.mean([i.mean() for _, i in cells]):.3f}",
                              f"{np.mean([b.mean() for b, _ in cells]):.3f}", ci(r)])
        L += [f"### {COND_DESC[cd]}：依構型組彙總", "",
              table(["構型組", "任務", "informed 分數", "最強古典法分數", "不知道的代價 pp [95% CI]"], grows), ""]
    # 事實陳述
    worse = [(cd, c, t, r) for cd in CONDS for c in CONFIGS for t in TASKS
             for r in [boot_pp([(sc_arr(g2, cd, c, best_classical(g2, cd, c, t), t), np.array(inf[key(cd, c, NAME, t)]["score"], float))], rng)]
             if r[0] < 0]
    L += ["## 事實陳述（只列數字）", ""]
    for cd in CONDS:
        parts = [f"{g.split('（')[0]} {summary[(cd, g, '兩任務平均')][0]:.1f} pp（POS_STEP {summary[(cd, g, 'POS_STEP')][0]:.1f}、T2 {summary[(cd, g, 'T2')][0]:.1f}）"
                 for g in GROUPS]
        L.append(f"- {COND_DESC[cd]}：{'；'.join(parts)}。")
    nbig = sum(1 for cd in CONDS for c in CONFIGS for t in TASKS if
               pp(sc_arr(g2, cd, c, best_classical(g2, cd, c, t), t).mean(), np.mean(inf[key(cd, c, NAME, t)]["score"])) >= 5)
    L.append(f"- 36 格（3 條件 × 6 構型 × 2 任務）中，代價 ≥ 5 pp 的有 {nbig} 格；informed 比最強古典法差（代價 < 0）的有 {len(worse)} 格"
             + ("：" + "、".join(f"{cd}/{c}/{t} {r[0]:.1f}（CI [{r[1]:.1f}, {r[2]:.1f}]）" for cd, c, t, r in worse) if worse else "") + "。")
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eps", type=int, default=100)
    ap.add_argument("--tag", default="")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    g2 = json.loads((RES / "g2_raw.json").read_text())
    path = RES / f"g2_informed_raw{args.tag}.json"
    inf = json.loads(path.read_text()) if args.report_only else run_all(args, path)
    if args.n_eps != 100:
        print("n_eps != 100：與 g2_raw 不成對，不寫正式報告"); return
    report(g2, inf, RES / f"g2_informed{args.tag}.md", args)
    print("已寫入", RES / f"g2_informed{args.tag}.md")


if __name__ == "__main__":
    main()
