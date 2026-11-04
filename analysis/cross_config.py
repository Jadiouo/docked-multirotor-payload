"""P0c：跨構型模糊。policy 共享權重，所以「k 跳視野相同」的模組不論屬於哪個構型，輸出都必須相同。
線性共享分配：視野類別 v → 4×3 矩陣 W_v，模組指令 u_i = W_v(i)·α_des（只看差動部分，不含推力上下限）；
構型 c 實得角加速度 G_c·α_des，G_c = J_c⁻¹ Σ_i A_i W_v(i)（A_i = 模組 i 的 3×4 力矩子矩陣）。
執行：.venv/bin/python -m analysis.cross_config"""
import numpy as np
from quadsim.assembly import CONFIGS, TRAIN, assemble
from analysis.khop import neighbors, refinements

KS = ["盲", 0, 1, 2, 3, "∞"]
I3 = np.eye(3)
S3 = np.sqrt(3)                       # 「完全沒控制力」的殘差基準 ||0 − I||
ASM = {n: assemble(c) for n, c in CONFIGS.items()}
NBR = {n: neighbors(c) for n, c in CONFIGS.items()}


def view_classes(k):
    """回傳 {構型: [每個模組的類別編號]}。顏色用巢狀 tuple（內容本身），所以跨構型可直接比較；
    '盲' = 全部同色；'∞' = 每個 (構型, 模組) 各自一類。"""
    ids, out = {}, {}
    for n, nbr in NBR.items():
        N = len(nbr)
        if k == "盲": cols = [0] * N
        elif k == "∞": cols = [(n, i) for i in range(N)]
        else:
            cols = [tuple(j is not None for j in row) for row in nbr]          # k=0：四邊有沒有鄰居
            for _ in range(k):                                                 # 每輪：(舊顏色, 四方向鄰居的舊顏色)
                cols = [(cols[i], tuple(None if j is None else cols[j] for j in nbr[i])) for i in range(N)]
        out[n] = [ids.setdefault(c, len(ids)) for c in cols]
    return out, len(ids)


def blocks(n, cls, V):
    """B[v] = J⁻¹ Σ_{i 屬於類別 v} A_i（3×4），使 G_c = Σ_v B[v] W_v 對 W 是線性的。"""
    s = assemble(CONFIGS[n]); Ji = np.linalg.inv(s["J"]); B = np.zeros((V, 3, 4))
    for i, v in enumerate(cls[n]): B[v] += Ji @ s["A"][1:, 4 * i:4 * i + 4]
    return B


def fit(fit_names, cls, V):
    """最小二乘求 W：讓 fit_names 內每個構型的 G_c 盡量接近 I（欠定時取最小範數解）。
    回傳 (W, seen)：seen = 擬合集裡出現過的類別；沒見過的類別沒有 W，由 run() 用降一級 k 的結果頂替。"""
    Bs = {n: blocks(n, cls, V) for n in fit_names}
    seen = sorted({v for n in fit_names for v in cls[n]})
    M = np.vstack([np.hstack([Bs[n][v] for v in seen]) for n in fit_names])      # (3C, 4|seen|)
    sol = np.linalg.lstsq(M, np.vstack([I3] * len(fit_names)), rcond=None)[0]
    W = np.zeros((V, 4, 3)); W[seen] = sol.reshape(len(seen), 4, 3)
    return W, set(seen)


def score(G):
    """(G, a, b)。a = ‖G−I‖_F。b = 逐軸對角適應 ‖G·diag(1/g_k) − I‖_F，g_k = G 的對角元素（每軸各自補回增益）。
    g_k ≤ 0 代表該軸沒有控制力（甚至反向），適應不了；適應後殘差若超過「完全沒控制力」的 √3 基準，
    也視為無法適應。兩種情況 b 都記 nan（表中顯示「無法」），不再除下去。"""
    g = np.diag(G)
    b = np.linalg.norm(G / g - I3) if np.all(g > 1e-6) else np.nan      # G/g 逐欄除以 g_k = G·diag(1/g)
    return G, np.linalg.norm(G - I3), (b if b <= S3 else np.nan)


def residuals(n, Wmod):
    """Wmod[i] = 模組 i 實際用的 W（4×3）。G_c = J⁻¹ Σ A_i W_i。"""
    s_ = ASM[n]
    return score(np.linalg.inv(s_["J"]) @ sum(s_["A"][1:, 4 * i:4 * i + 4] @ w for i, w in enumerate(Wmod)))


def run(fit_names):
    """對每個 k 擬合並評估全部構型。回傳 {k: {構型: (G, a, b, 沒見過類別的模組比例)}}。
    沒見過的類別 fallback 到「降一級 k」該模組實際使用的 W（一路降到盲；盲是全域單一類別，一定見過）。"""
    res, prev = {}, None
    for k in KS:
        cls, V = view_classes(k); W, seen = fit(fit_names, cls, V)
        cur, res[k] = {}, {}
        for n in CONFIGS:
            un = [v not in seen for v in cls[n]]
            cur[n] = [prev[n][i] if u else W[v] for i, (v, u) in enumerate(zip(cls[n], un))]
            res[k][n] = residuals(n, cur[n]) + (float(np.mean(un)),)
        prev = cur
    return res


def hop_dist(n):
    """構型 n 的模組兩兩圖距離（BFS，docking graph 四鄰）。"""
    nbr, N = NBR[n], len(NBR[n]); D = np.full((N, N), 10**6)
    for a in range(N):
        D[a, a] = 0; q = [a]
        for u in q:
            for v in nbr[u]:
                if v is not None and D[a, v] > D[a, u] + 1: D[a, v] = D[a, u] + 1; q.append(v)
    return D


def radius_of(k):
    """k 對應的球半徑：WL 第 k 輪（k=0 是鄰居旗標）已涵蓋距離 k+1 的格點佔用資訊；盲 = 只有自己；∞ = 整架。"""
    return 0 if k == "盲" else (10**6 if k == "∞" else k + 1)


def ball_xy(n, i, r):
    """模組 i 的 r 跳球內，所有模組相對 i 的格點座標（含自己）。"""
    c = np.array(CONFIGS[n]); m = hop_dist(n)[i] <= r
    return c[m] - c[i]


def phi(n, i, r):
    """結構式局部特徵 φ_i：球內相對自己的質心偏移 (cx, cy)（格點單位），及 +x、−x、+y、−y 四個方向各有幾個模組。"""
    d = ball_xy(n, i, r)
    return np.array([d[:, 0].mean(), d[:, 1].mean(), (d[:, 0] > 0).sum(), (d[:, 0] < 0).sum(), (d[:, 1] > 0).sum(), (d[:, 1] < 0).sum()], float)


def struct_C(n, r):
    """G_c = Σ_p C[p] W_p，p=0 是 W0，p=1..6 對應 6 個特徵。W_i = W0 + Σ_j φ_ij W_j → 對 W 線性，參數量 7×4×3 與 k 無關。"""
    s_ = ASM[n]; Ji = np.linalg.inv(s_["J"]); Ai = [s_["A"][1:, 4 * i:4 * i + 4] for i in range(s_["N"])]
    F = np.array([phi(n, i, r) for i in range(s_["N"])])
    return np.array([Ji @ sum(Ai)] + [Ji @ sum(F[i, j] * Ai[i] for i in range(s_["N"])) for j in range(6)])


def run_struct(fit_names):
    """結構式：用 fit_names 擬合 W0..W6，評估全部構型。回傳 {k: {構型: (G, a, b)}}。"""
    res = {}
    for k in KS:
        r = radius_of(k); C = {n: struct_C(n, r) for n in CONFIGS}
        M = np.vstack([np.hstack(C[n]) for n in fit_names])                     # (3C, 28)
        W = np.linalg.lstsq(M, np.vstack([I3] * len(fit_names)), rcond=None)[0].reshape(7, 4, 3)
        res[k] = {n: score(sum(c @ w for c, w in zip(C[n], W))) for n in CONFIGS}
    return res


def wl_vs_geometry(ks=("盲", 0, 1, 2, 3)):
    """交叉檢查：WL 分割 vs「r 跳球內相對座標集合」相等分割（在所有構型的聯集上）。回傳 {k: 不一致描述 list}。"""
    out = {}
    for k in ks:
        cls, _ = view_classes(k); r = radius_of(k)
        key = {(n, i): frozenset(map(tuple, ball_xy(n, i, r))) for n in CONFIGS for i in range(len(NBR[n]))}
        wl = {(n, i): cls[n][i] for n in CONFIGS for i in range(len(NBR[n]))}
        bad = []
        for a in key:                                  # 兩種分割一致 ⇔ 互相對應成一對一
            for b in key:
                if a < b and (key[a] == key[b]) != (wl[a] == wl[b]):
                    bad.append((a, b, "幾何相同 WL不同" if key[a] == key[b] else "WL相同 幾何不同"))
        out[k] = bad
    return out


def fmt(r):
    return "%.2f / %s" % (r[1], "無法" if np.isnan(r[2]) else "%.2f" % r[2])


def agg(rs, f):
    """a、b 的平均／最差；b 為 nan（無法適應）時以 √3 計入，並標出個數。"""
    a = f([r[1] for r in rs]); bs = [S3 if np.isnan(r[2]) else r[2] for r in rs]
    return "%.2f / %.2f%s" % (a, f(bs), "（%d 個無法）" % sum(np.isnan(r[2]) for r in rs) if any(np.isnan(r[2]) for r in rs) else "")


LAB = {"盲": "盲", 0: "0（鄰居旗標）", 1: "1", 2: "2", 3: "3", "∞": "∞"}

if __name__ == "__main__":
    held = [n for n in CONFIGS if n not in TRAIN]
    R = {"train": run(TRAIN), "all": run(list(CONFIGS))}
    S = run_struct(TRAIN); S_all = run_struct(list(CONFIGS))
    hdr = "| 構型 | " + " | ".join(f"k={LAB[k]}" for k in KS) + " |\n|---|" + "---|" * len(KS)
    md = ["# P0c 跨構型模糊（共享 policy）\n",
          "模型：視野類別 v → W_v（4×3），u_i = W_v(i)·α_des，G_c = J_c⁻¹ Σ A_i W_v(i)；目標 G_c ≈ I。只看差動力矩，**線性模型不含推力上下限（0≤T≤Tmax）**，所以是樂觀的。",
          "殘差 a = ‖G−I‖_F（完全沒控制力 = √3 = 1.73）；b = 逐軸對角適應 ‖G·diag(1/g_k)−I‖_F，g_k = G 的對角元素（每軸各自補回增益，代表有歷史的 policy）。"
          "g_k ≤ 0（該軸沒有控制力或反向）或適應後殘差 >√3 時記「無法」，不再除下去。",
          "**命名**：「k=0（鄰居旗標）」= 只知道自己四邊有沒有鄰居；物理上它已經涵蓋半徑 1 的格點佔用資訊，WL 第 k 輪涵蓋半徑 k+1。",
          "**k 與直徑／半徑**：線形構型的穩定 k = 半徑 − 1（line2、line3 為 0，line4 為 1，line6 為 2，line8 為 3）；格狀與其他二維構型在 k=0 就穩定。",
          "不報單一門檻 k*，改報每個 k 的完整曲線（以下各表）。\n"]
    md += ["## A. 查表（每個視野類別一個 W），只用 TRAIN 擬合，格內 a / b\n",
           "沒見過的視野類別沒有 W，改用「降一級 k」時該模組實際使用的 W，一路降到盲（盲是單一全域類別，一定見過）。k=∞ 在留出構型上全部沒見過。\n", hdr]
    for n in CONFIGS: md.append(f"| {n}{' (留出)' if n in held else ''} | " + " | ".join(fmt(R['train'][k][n]) for k in KS) + " |")
    for lab, f in (("平均", np.mean), ("最差", np.max)): md.append(f"| **{lab}（留出）** | " + " | ".join(agg([R['train'][k][n] for n in held], f) for k in KS) + " |")
    md += ["", "留出構型中「沒見過類別」的模組比例（k 越大越高 → 大 k 泛化變差的訊號）：\n", hdr]
    md += [f"| {n} | " + " | ".join("%.0f%%" % (100 * R['train'][k][n][3]) for k in KS) + " |" for n in held]
    md += ["", "## B. 結構式（W_i = W0 + Σ_j φ_ij W_j，7×4×3 個參數，與 k 無關），只用 TRAIN 擬合\n",
           "φ_i = k 跳球（半徑 k+1）內相對自己的質心偏移 (cx, cy)＋四個方向的模組數。k=∞ 欄 = 球涵蓋整架，即使用真實質心的結構式。\n", hdr]
    for n in CONFIGS: md.append(f"| {n}{' (留出)' if n in held else ''} | " + " | ".join(fmt(S[k][n]) for k in KS) + " |")
    for lab, f in (("平均", np.mean), ("最差", np.max)): md.append(f"| **{lab}（留出）** | " + " | ".join(agg([S[k][n] for n in held], f) for k in KS) + " |")
    ma = [np.mean([S[k][n][1] for n in held]) for k in KS]
    mono = all(x >= y - 1e-9 for x, y in zip(ma, ma[1:]))
    md += ["", f"結構式留出構型平均 a 隨 k 是否單調下降：**{'是' if mono else '否'}**（" + " → ".join("%.2f" % v for v in ma) + "）。\n",
           "## C. 三種函數類別並列（留出構型平均 a / b；結構式 k=∞ 為真實質心版本）\n", hdr.replace("構型", "函數類別")]
    md.append("| 查表 | " + " | ".join(agg([R['train'][k][n] for n in held], np.mean) for k in KS) + " |")
    md.append("| 結構式 | " + " | ".join(agg([S[k][n] for n in held], np.mean) for k in KS) + " |")
    md.append("| 結構式（真實質心，k=∞，與 k 無關） | " + " | ".join(agg([S['∞'][n] for n in held], np.mean) for k in KS) + " |")
    md += ["", "## D. 查表，用全部構型擬合（容量上限），格內 a / b\n", hdr]
    for n in CONFIGS: md.append(f"| {n} | " + " | ".join(fmt(R['all'][k][n]) for k in KS) + " |")
    md.append("| **平均** | " + " | ".join(agg([R['all'][k][n] for n in CONFIGS], np.mean) for k in KS) + " |")
    md.append("| **最差** | " + " | ".join(agg([R['all'][k][n] for n in CONFIGS], np.max) for k in KS) + " |")
    neq = 9 * len(CONFIGS); npar = []
    for k in KS:
        cls, V = view_classes(k); npar.append(12 * V)
    md += ["", f"**k≥2 殘差為 0 的原因**：查表的參數數 = 類別數 × 12，方程數 = 構型數 × 9 = {neq}；參數 ≥ 方程數時 lstsq 可以直接記住答案，量到的是記憶容量，不是真正的表示能力。\n",
           "| k | " + " | ".join(str(LAB[k]) for k in KS) + " |", "|---|" + "---|" * len(KS),
           "| 參數數 | " + " | ".join(str(x) for x in npar) + " |", "| 參數數 / 方程數 | " + " | ".join("%.2f" % (x / neq) for x in npar) + " |",
           f"| 結構式（固定 84 參數）/ {neq} | " + " | ".join("%.2f" % (84 / neq) for _ in KS) + " |"]
    md += ["", "## E. 增益偏差 vs 軸間耦合（查表、全部構型擬合，k=0 與 k=1）\n", "| k | 構型 | 對角元素 | 非對角最大值 |", "|---|---|---|---|"]
    for k in (0, 1):
        for n in ("line4", "line8", "grid10"):
            G = R["all"][k][n][0]; md.append(f"| {k} | {n} | " + "/".join("%.2f" % v for v in np.diag(G)) + f" | {np.abs(G - np.diag(np.diag(G))).max():.2f} |")
    X = wl_vs_geometry(); md += ["", "## F. 交叉檢查：1-WL 分割 vs「球內相對座標集合」分割（所有構型聯集）\n",
                                 "球半徑 = k+1（盲 = 0）。兩種分割一致 ⇔ 同一對模組要嘛都相同要嘛都不同。\n", "| k | 不一致的模組對數 | 例 |", "|---|---|---|"]
    for k, bad in X.items(): md.append(f"| {LAB[k]} | {len(bad)} | {bad[:2] if bad else '—'} |")
    md += ["", "## 解讀\n", """1. 逐軸增益適應（b）幾乎吞掉所有殘差（留出平均 b 0.03–0.08），因為非對角（軸間耦合）≈ 0（L3 除外，b 仍 0.25–0.74）；剩下的主要是「每軸增益偏差」，這正是歷史／適應能補的部分，a 才是不適應時的真實風險。
2. 查表在留出構型上沒有隨 k 變好：k=1 起大半視野沒見過，fallback 後平均 a 停在 0.93，且沒見過比例隨 k 升到 100%；全部構型擬合在 k≥2 殘差為 0，只是參數數／方程數 ≥ 5.7 的記憶容量。
3. 結構式（84 個參數，與 k 無關）在留出構型上明顯優於查表：平均 a 從 1.45（盲）降到 0.83（k=0）、0.48（k=1），但之後約停在 0.45–0.47，**不隨 k 單調下降**（grid6、grid9、grid10 在 k 增大時反而變差）。
4. 預期「k=∞ 真實質心的結構式接近 pseudoinverse、殘差很小」只部分成立：平均 a 仍有 0.46（grid10 達 1.19），但適應後 b 的平均 0.03；原因是 pseudoinverse 的增益含 1/J（對構型非線性），線性特徵 (cx, cy, 計數) 表示不了，要靠適應補回增益。
5. 1-WL 分割與「半徑 k+1 球內相對座標集合」分割在所有構型聯集上、所有 k 完全一致，因此以 WL 為準；全部結果是線性、無推力上下限的樂觀估計。\n"""]
    open("results/p0/cross_config.md", "w").write("\n".join(md))
    print("\n".join(md))
