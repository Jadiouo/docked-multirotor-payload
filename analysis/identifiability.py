"""P0 可辨識性：symmetric 制度下「模組指令 → 機身角加速度」的增益 G 隨構型的變化，
以及含馬達延遲、陀螺雜訊／偏置、閉迴路時，用最小二乘估 g 到 ±10% 需要多久資料。
執行：.venv/bin/python -m analysis.identifiability"""
import numpy as np
from scipy.signal import lfilter
from quadsim.params import CF2X
from quadsim.assembly import P0_CONFIGS, TRAIN, assemble, sym_matrix
from analysis.controllability import lp_alpha

FS, DT, EXC = 200.0, 1 / 200.0, 0.1       # 取樣率、週期、激勵幅度（單位指令：1 = 單模組該軸最大差動力矩）
BETA = 1 - np.exp(-DT / CF2X["tau"])      # 一階馬達延遲的離散化係數（tau = 0.07 s）
GYRO = {"A (0.005 無偏置)": (0.005, 0.0), "B (0.02 偏置0.01)": (0.02, 0.01)}
N_MAX, TRIALS = 12000, 200                # 最長 60 s
AX = ["roll", "pitch", "yaw"]

# 單模組局部最大力矩 τ_local（各軸 ± 取小）= 單模組 α 權限 × 其 J
S = assemble(P0_CONFIGS["single"])
tau_loc = np.array([min(lp_alpha(S["A"], S["M"], S["J"], k, +1), lp_alpha(S["A"], S["M"], S["J"], k, -1)) * S["J"][k, k] for k in range(3)])
# G = J⁻¹ · N·diag(τ_local)；symmetric 力矩恆等於 N 倍局部力矩（力臂項總和為 0），assert 驗證
Gm = {}
for name, cells in P0_CONFIGS.items():
    s = assemble(cells)
    assert np.allclose(sym_matrix(s)[1:], s["N"] * sym_matrix(S)[1:])
    Gm[name] = np.linalg.inv(s["J"]) @ (s["N"] * np.diag(tau_loc))
g = {n: np.diag(m) for n, m in Gm.items()}     # 各軸有效增益（對角）；L3 的耦合另外用奇異值報告


def first_ok(ghat, g_true):
    """ghat: (trials, n)。回傳「95% 試驗誤差 <10%」的最短毫秒數，達不到回傳 inf。"""
    ok = np.mean(np.abs(ghat / g_true - 1) < 0.10, axis=0) >= 0.95
    ok[:2] = False
    ok = np.logical_and.accumulate(ok[::-1])[::-1]       # 從這個時間點之後都持續達標才算
    return (np.argmax(ok) + 1) * DT * 1e3 if ok.any() else np.inf


def open_loop(g_true, sigma, bias, rng, wrong=False):
    """開迴路：u 白雜訊 → 馬達延遲 x → ω。回歸 α_meas = g·x（x = 延遲後的實際推力，模組自己算得出）。
    wrong=True：誤用同一時刻指令 u 當回歸量（pitfall）。回傳 (需時 ms, 最終偏誤)。"""
    u = rng.uniform(-EXC, EXC, (TRIALS, N_MAX))
    x = lfilter([0, BETA], [1, -(1 - BETA)], u, axis=1)               # x_k = (1-β)x_{k-1} + β u_{k-1}
    w = np.c_[np.zeros(TRIALS), np.cumsum(g_true * x * DT, axis=1)] + bias + rng.normal(0, sigma, (TRIALS, N_MAX + 1))
    a = np.diff(w, axis=1) / DT                                        # 常數偏置在差分時消掉
    r = u if wrong else x
    ghat = np.cumsum(r * a, 1) / np.cumsum(r * r, 1)
    return first_ok(ghat, g_true), np.mean(ghat[:, -1]) / g_true - 1


def closed_loop(g_true, sigma, bias, rng, wn=8.0, zeta=0.9):
    """閉迴路：PD 姿態控制（增益依真實 g 縮放）+ 小 dither，回歸量 x 同上。
    回饋讓 u 與雜訊／狀態相關，檢查估計偏誤。回傳 5 s 與 60 s 的平均偏誤。"""
    th = np.zeros(TRIALS); w = np.zeros(TRIALS); x = np.zeros(TRIALS)
    Sxa = np.zeros(TRIALS); Sxx = np.zeros(TRIALS); out = {}
    wm = w + bias + rng.normal(0, sigma, TRIALS)
    for k in range(N_MAX):
        u = np.clip((-wn**2 * th - 2 * zeta * wn * wm) / g_true + rng.uniform(-EXC, EXC, TRIALS), -1, 1)
        x_k = x.copy()
        x += BETA * (u - x)
        w = w + g_true * x_k * DT; th = th + w * DT
        wm2 = w + bias + rng.normal(0, sigma, TRIALS)
        Sxa += x_k * (wm2 - wm) / DT; Sxx += x_k * x_k; wm = wm2
        if k + 1 in (1000, N_MAX): out[(k + 1) * DT] = np.mean(Sxa / Sxx) / g_true - 1
    return out


def imu_note():
    """line4 兩端模組的加速度計差異：a = α×r + ω×(ω×r)（共同的重力與線加速度在兩端相同，相減消去）。"""
    r = assemble(P0_CONFIGS["line4"])["rotor_pos"]; c = r.reshape(4, 4, 3).mean(1)   # 4 個模組中心相對 CoM
    al, om = np.array([0, 50.0, 0]), np.array([0, 5.0, 0])
    acc = lambda rr: np.cross(al, rr) + np.cross(om, np.cross(om, rr))
    d = acc(c[-1]) - acc(c[0])
    return c[-1][0], np.linalg.norm(d), np.linalg.norm(d) / 0.05, np.linalg.norm(d) / (0.05 * np.sqrt(2))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    out = ["# P0 可辨識性\n",
           "G = J⁻¹·N·diag(τ_local)，單位 rad/s² 每單位指令；對角元素 = 各軸有效增益 g。",
           "馬達延遲 τ=0.07 s（離散化 β=%.3f）；回歸量 = 延遲後的實際推力（模組知道馬達模型與自己的指令）。" % BETA,
           "判準：%d 次試驗中 95%% 的估計誤差 < 10%%，且此後持續達標；資料最長 60 s（inf = 60 s 內達不到）。\n" % TRIALS,
           "## 增益 g 與增益範圍\n", "| 構型 | g_roll | g_pitch | g_yaw |", "|---|---|---|---|"]
    out += [f"| {n} | " + " | ".join("%.0f" % v for v in gv) + " |" for n, gv in g.items()]
    for label, names in (("訓練構型", TRAIN), ("全部構型", list(P0_CONFIGS))):
        G = np.array([g[n] for n in names]); out.append(f"\n{label} 最大/最小比值：" + "，".join(f"{a} {v:.1f}" for a, v in zip(AX, G.max(0) / G.min(0))))
    sv = np.linalg.svd(Gm["L3"], compute_uv=False)
    out.append("\nL3 的 G 矩陣（有 roll/pitch 耦合）：\n```\n%s\n```\n奇異值 %s（最大/最小 = %.1f）；其餘構型 J 為對角，G 即對角矩陣。"
               % (np.array2string(Gm["L3"], precision=0, suppress_small=True), np.round(sv), sv[0] / sv[-1]))
    for gname, (sig, b) in GYRO.items():
        out += [f"\n## 陀螺儀 {gname}：開迴路，估到 ±10% 所需毫秒（roll/pitch/yaw）\n", "| 構型 | 正確回歸 x | 錯誤回歸 u：同刻指令的偏誤 roll/pitch/yaw |", "|---|---|---|"]
        for n in P0_CONFIGS:
            T = [open_loop(g[n][i], sig, b, rng)[0] for i in range(3)]
            W = [open_loop(g[n][i], sig, b, rng, wrong=True)[1] for i in range(3)]
            out.append(f"| {n} | " + "/".join("%.0f" % t for t in T) + " | " + "/".join("%+.0f%%" % (100 * v) for v in W) + " |")
    out += ["\n白雜訊指令與延遲後推力的互相關為 0，所以誤用同刻指令回歸時 ĝ→0（偏誤 ≈ -100%）。"
            "B 組的常數偏置 0.01 rad/s 在差分時完全消掉，所以 B 組變慢只來自雜訊 4 倍；緩慢漂移的偏置才會進入估計，本分析沒有模擬。",
            "\n## 閉迴路（PD wn=8, ζ=0.9 + dither ±0.1）：估計 ĝ/g − 1 的平均偏誤\n\n近 0 的偏誤有一部分是結構性的：回歸量 x_k 只含過去的指令，與 α_k 的陀螺雜訊在模擬中結構上不相關；真實陀螺有低通濾波，雜訊會有時間相關，偏誤可能變大。\n", "| 案例 | 陀螺 | 5 s | 60 s |", "|---|---|---|---|"]
    for n, i in (("single", 0), ("square4", 1), ("line4", 1)):
        for gname, (sig, b) in GYRO.items():
            o = closed_loop(g[n][i], sig, b, rng)
            out.append(f"| {n} {AX[i]} | {gname} | " + " | ".join("%+.1f%%" % (100 * v) for v in o.values()) + " |")
    x, d, r1, r2 = imu_note()
    out += ["\n## IMU 打破對稱（line4 兩端模組，α=50 rad/s²，ω=5 rad/s，繞 pitch 軸）\n",
            f"兩端模組距 CoM ±{x:.2f} m，兩端加速度計讀數**相減**的大小 = {d:.1f} m/s²（α×r 項 15 m/s² 垂直，向心項 7.5 m/s² 水平）；"
            f"對加速度計雜訊 0.05 m/s²（**假設值**）：差/雜訊 = {r1:.0f}（差值雜訊取 √2 倍則 {r2:.0f}）。",
            "限制：這個比值必須同時拿到兩個模組的讀數才算得出來，單一模組做不到。單一模組自己的讀數含未知的 CoM 線加速度，"
            "而且不知道 N，所以扣不掉；就算扣掉，旋轉項也只能給出 |r|，得不到方向。因此不能說單一模組可以從自己的 IMU 推出位置；"
            "此數字只說明「若有跨模組比較（通訊或共用歷史）才有的資訊量」，靜止 hover（α≈0, ω≈0）時兩端讀數相同，完全沒有資訊。"]
    open("results/p0/identifiability.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))
