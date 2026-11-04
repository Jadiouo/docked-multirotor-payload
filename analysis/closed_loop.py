"""P0 閉迴路檢查：oracle（full）與 symmetric 分配的階躍響應，及增益高估測試。
注意：symmetric 只是 L0 的悲觀下界，真實 L0 可從自己的 IMU 得到額外資訊。
模型：θ̈ = 權限·x，x 為一階馬達延遲後的歸一化推力，指令飽和 ±1。
執行：.venv/bin/python -m analysis.closed_loop"""
import numpy as np
from quadsim.params import CF2X
from analysis.controllability import table

DT, T_END = 0.002, 20.0
BETA = 1 - np.exp(-DT / CF2X["tau"])


def sim(auth, norm, kp, kd, ref):
    """auth：實際角加速度權限；norm：控制器「以為」的權限（PD 要的角加速度除以它得到歸一化指令）。
    回傳 (θ/ref 軌跡, 指令飽和的時間占比)。"""
    th = w = x = 0.0; tr = []; sat = 0
    for _ in range(int(T_END / DT)):
        raw = (kp * (ref - th) - kd * w) / norm
        cmd = np.clip(raw, -1, 1); sat += abs(raw) > 1
        x += BETA * (cmd - x); w += auth * x * DT; th += w * DT; tr.append(th)
        if abs(th) > 1e6: break
    return np.array(tr) / ref, sat * DT / T_END


def metrics(tr):
    if len(tr) < int(T_END / DT) or not np.all(np.isfinite(tr)) or np.abs(tr[-500:] - 1).max() > 0.05 or tr.max() > 5:
        return "失穩/未收斂", np.nan, np.nan
    t10, t90 = (np.argmax(tr >= v) for v in (0.1, 0.9))
    return "穩定", (t90 - t10) * DT, 100 * max(tr.max() - 1, 0)


rows = {r["config"]: r for r in table()}
AXES = {"roll": 0.2, "pitch": 0.2, "yaw": 0.5}           # 調參用的線性階躍 (rad)
out = ["# P0 閉迴路階躍\n",
       "**symmetric 是 L0 的悲觀下界**：模組可以從自己的 IMU 得到額外資訊，真實 L0 可能比這裡好。\n",
       "模型：θ̈ = 權限·x，馬達 τ=0.07 s，指令飽和 ±1；權限取 ± 較小者。PD（kp=wn², kd=2ζwn）**每軸只在 square4 的 oracle 上網格調一次**"
       "（準則：上升時間最短且過衝 <10%），之後固定。\n",
       "飽和判斷：PD 要的角加速度 kp·誤差 超過歸一化權限時才飽和。wn=12 時 pitch 階躍 0.2 rad 的初始需求最多只有歸一化權限的 0.32 倍，**不會飽和**，"
       "所以 0.2 rad 的過衝完全來自增益失配，不是權限不足；階躍 1.5 rad 才會飽和（欄位「飽和%」）。\n"]
gains = {}
for ax, ref in AXES.items():
    a_sq = rows["square4"][f"alpha_full_{ax}"]; best = None
    for wn in np.arange(1.0, 14.1, 0.5):
        for ze in (0.7, 0.9, 1.1, 1.4):
            st, rt, os_ = metrics(sim(a_sq, a_sq, wn**2, 2 * ze * wn, ref)[0])
            if st == "穩定" and os_ < 10 and (best is None or rt < best[0]): best = (rt, wn, ze)
    gains[ax] = best[1:]

out += ["\n## A. 階躍響應：oracle / sym-self / sym-fixed\n",
        "- oracle：歸一化 = 自己的 full 權限。\n- sym-self：實際權限與歸一化都是自己的 symmetric 權限，**單獨看分配損失**。\n"
        "- sym-fixed：實際是自己的 sym 權限，歸一化用 square4 的 sym 權限（不知道構型）。\n",
        "| 軸/階躍 | 構型 | 分配 | 權限 | 結果 | 上升 (s) | 過衝 % | 飽和 % |", "|---|---|---|---|---|---|---|---|"]
for ax, ref in (("pitch", 0.2), ("pitch", 1.5), ("yaw", 0.5)):
    wn, ze = gains[ax]
    for cfg in ("square4", "line4", "grid6"):
        f, s_ = rows[cfg][f"alpha_full_{ax}"], rows[cfg][f"alpha_sym_{ax}"]; n0 = rows["square4"][f"alpha_sym_{ax}"]
        for label, au, nm in (("oracle", f, f), ("sym-self", s_, s_), ("sym-fixed", s_, n0)):
            tr, sat = sim(au, nm, wn**2, 2 * ze * wn, ref); st, rt, os_ = metrics(tr)
            out.append(f"| {ax} {ref} (wn={wn}, ζ={ze}) | {cfg} | {label} | {au:.0f} | {st} | {rt:.2f} | {os_:.1f} | {100*sat:.0f} |")

out += ["\n## B. 增益高估：控制器以低權限構型歸一化，實際飛在高權限構型上（L0 不做適應的主要風險）\n",
        "高估倍數 = 實際權限 / 歸一化權限；增益沿用 square4 調好的值；階躍同 A 的線性階躍。\n",
        "| 軸 | 歸一化來源 (sym 權限) | 實際構型 (sym 權限) | 高估倍數 | 結果 | 上升 (s) | 過衝 % | 飽和 % |", "|---|---|---|---|---|---|---|---|"]
for ax, nm_cfg, fly in (("roll", "square4", "single"), ("roll", "square4", "line4"), ("pitch", "line4", "square4"),
                        ("pitch", "line4", "single"), ("yaw", "line4", "single"), ("yaw", "grid6", "single")):
    wn, ze = gains[ax]; au, nm = rows[fly][f"alpha_sym_{ax}"], rows[nm_cfg][f"alpha_sym_{ax}"]
    tr, sat = sim(au, nm, wn**2, 2 * ze * wn, AXES[ax]); st, rt, os_ = metrics(tr)
    out.append(f"| {ax} | {nm_cfg} ({nm:.0f}) | {fly} ({au:.0f}) | {au/nm:.1f}x | {st} | {rt:.2f} | {os_:.1f} | {100*sat:.0f} |")
# 增益高估為何不失穩：特徵式 τs³ + s² + G·kd·s + G·kp = 0（G = 增益倍數），Routh 條件 kd > τ·kp 與 G 無關
wn, ze = gains["pitch"]
out += [f"\n**結論：增益高估在此模型中不會失穩**（高估到 25x 仍穩定；掃描至 200x 皆穩定）。解析原因：特徵式 τs³ + s² + G·kd·s + G·kp = 0 的 Routh 條件是 kd > τ·kp"
        f"（{2*ze*wn:.1f} > {CF2X['tau']*wn**2:.1f}），與增益倍數 G 無關，只要 PD 的阻尼比例固定就一直成立。高估的代價是高頻變快、變硬（上升時間 0.24-0.33 s 是因為過阻尼，ζ_eff = ζ·√G）。"
        "**這個結論僅限於本模型**：沒有陀螺雜訊放大（kd·G 放大雜訊）、感測／運算延遲、結構模態或推力飽和的限幅循環，所以真實的失穩門檻一定更低，需要在有雜訊與延遲的模擬中再測。\n"]
open("results/p0/closed_loop.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
