"""P0 可控性分析：各構型在 full / symmetric 兩種分配制度下的角加速度權限（α = J⁻¹τ）。
執行：.venv/bin/python -m analysis.controllability"""
import csv, os
import numpy as np
from scipy.optimize import linprog
from quadsim.params import CF2X, G
from quadsim.assembly import P0_CONFIGS, assemble, sym_matrix

AXES = ["roll", "pitch", "yaw"]
TMAX = CF2X["Tmax"]


def lp_alpha(Ar, M, J, k, sgn):
    """Fz = Mg、另外兩軸角加速度 = 0、0≤T≤Tmax 下，最大化 sgn·α_k（α = J⁻¹τ）。
    用 α 而不是 τ，因為 J 有非對角項（L3）時，純 τ 不等於純角加速度。"""
    B = np.linalg.inv(J) @ Ar[1:]
    oth = [i for i in range(3) if i != k]
    r = linprog(-sgn * B[k], A_eq=np.vstack([Ar[0], B[oth]]), b_eq=[M * G, 0, 0], bounds=(0, TMAX), method="highs")
    return -r.fun if r.success else 0.0


def lp_both(Ar, M, J, s1, s2):
    """roll 與 pitch 同時需求：α_roll = s1·k、α_pitch = s2·k、α_yaw = 0，最大化 k。變數 = [T, k]。"""
    B = np.linalg.inv(J) @ Ar[1:]; n = Ar.shape[1]
    A_eq = np.vstack([np.r_[Ar[0], 0], np.r_[B[0], -s1], np.r_[B[1], -s2], np.r_[B[2], 0]])
    r = linprog(np.r_[np.zeros(n), -1], A_eq=A_eq, b_eq=[M * G, 0, 0, 0], bounds=[(0, TMAX)] * n + [(0, None)], method="highs")
    return -r.fun if r.success else 0.0


def yaw_free(Ar, M):
    """只限制 Fz = Mg，不限制 τx、τy 時的最大 τz（yaw 證明的上界）。"""
    r = linprog(-Ar[3], A_eq=Ar[:1], b_eq=[M * G], bounds=(0, TMAX), method="highs")
    return -r.fun


def table():
    rows = []
    for name, cells in P0_CONFIGS.items():
        s = assemble(cells); A, N, M, J = s["A"], s["N"], s["M"], s["J"]
        As = sym_matrix(s)
        row = dict(config=name, N=N, M=M, rank=np.linalg.matrix_rank(A),
                   sigma_min=np.linalg.svd(A, compute_uv=False)[-1], hover_frac=M * G / (4 * N * TMAX),
                   Jxy_over_Jxx=abs(J[0, 1]) / J[0, 0])
        for reg, Ar in (("full", A), ("sym", As)):
            for k, ax in enumerate(AXES):
                p, m = lp_alpha(Ar, M, J, k, +1), lp_alpha(Ar, M, J, k, -1)
                row[f"alpha_{reg}_+{ax}"], row[f"alpha_{reg}_-{ax}"] = p, m
                row[f"alpha_{reg}_{ax}"] = min(p, m)           # ± 取較小者（保守）
            row[f"both_{reg}"] = min(lp_both(Ar, M, J, a, b) for a in (1, -1) for b in (1, -1))
        for ax in AXES:
            row[f"ratio_{ax}"] = row[f"alpha_sym_{ax}"] / row[f"alpha_full_{ax}"]
        row["ratio_both"] = row["both_sym"] / row["both_full"]
        row["yaw_free_full"], row["yaw_free_sym"] = yaw_free(A, M), yaw_free(As, M)   # τz 上界（無 τx、τy 限制）
        T_h = np.full(4 * N, M * G / (4 * N))
        row["hover_net_torque"] = np.linalg.norm((A @ T_h)[1:])   # 全部推力相同時的淨力矩
        rows.append(row)
    return rows


PROOF = """## yaw 比值恆為 1 的證明

引理：單模組固定總推力 F 時，f(F) 的最大解是把推力集中在同向的對角兩顆馬達（轉向相同的 m0、m2，各 F/2，F ≤ 2Tmax；超過時另兩顆補餘量），此時 τx = a(T2−T0) = 0、τy = −a(T0−T2) = 0，所以模組內自動滿足 τx = τy = 0。\n\nyaw 力矩 τz = km·Σ s_k T_k 只取決於各轉子推力與轉向，和轉子位置無關。對單一模組，在該模組總推力 F 下可達的最大 yaw 是 F 的凹函數 f(F)（LP 的右端項參數化，分段線性）。
整架的 τz 上限 ≤ Σ_m f(F_m)，而 Σ F_m = Mg；由 Jensen 不等式，Σ f(F_m) ≤ N·f(Mg/N)，在總推力平均分配時取等號。
symmetric 解正好是每個模組 F = Mg/N，且各模組同樣輸出 τx = τy = 0 的最佳 yaw，因此它達到這個上界；τx = τy = 0 的約束只會讓 full 的上限更低，不會更高。故 τz,sym = τz,full，而 α_z = τz/Jzz 共用同一個 Jzz（平面構型 Jxz = Jyz = 0），比值 = 1。
數值核對（8 個構型）：無 τx、τy 限制的 τz 上界，full 與 sym 的最大差 = %.1e；表中 yaw 比值的最大偏離 1 = %.1e。無反例。
"""

if __name__ == "__main__":
    rows = table()
    os.makedirs("results/p0", exist_ok=True)
    with open("results/p0/controllability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    L = ["| 構型 | N | rank | σmin | hover/Tmax | α_full roll/pitch/yaw | α_sym roll/pitch/yaw | 比值 roll/pitch/yaw | 同時 k_full / k_sym / 比值 | Jxy/Jxx | hover 淨力矩 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f3 = lambda p, fmt="%.0f": "/".join(fmt % r[p + "_" + a] for a in AXES)
        L.append(f"| {r['config']} | {r['N']} | {r['rank']} | {r['sigma_min']:.3f} | {r['hover_frac']:.3f} | {f3('alpha_full')} | {f3('alpha_sym')} | "
                 f"{f3('ratio', '%.2f')} | {r['both_full']:.0f} / {r['both_sym']:.0f} / {r['ratio_both']:.2f} | {r['Jxy_over_Jxx']:.2f} | {r['hover_net_torque']:.0e} |")
    d1 = max(abs(r["yaw_free_full"] - r["yaw_free_sym"]) for r in rows)
    d2 = max(abs(r["ratio_yaw"] - 1) for r in rows)
    md = ("# P0 可控性\n\nα 單位 rad/s²，取 ± 方向較小者。單軸 = 其餘兩軸角加速度為 0；使用完整 J 的逆（含非對角項），α = J⁻¹τ。"
          "「同時」= α_roll = α_pitch = k、α_yaw = 0 的最大 k（四種正負號取最小）。\n\n"
          "**注意：Fz 固定為 Mg 會低估瞬間權限**（短時間可暫時增減總推力）。\n\n" + "\n".join(L) +
          "\n\n## 解讀\n\n1. 8 個構型的 A 秩都是 4，hover 時每顆推力 0.444 Tmax，全部可控，沒有要剔除的。\n2. 不知道位置（symmetric）損失的是力臂：square4 的 roll/pitch 下界 52%，grid6 pitch 下界 34%，line4 pitch 下界 26%，同時需求（roll=pitch）最差下界 25%（line4）。**注意 symmetric 只是 L0 的悲觀下界**：模組可以從自己的 IMU 取得額外資訊（見 identifiability.md），真實 L0 表現可能比這個好。\n3. 加入 J 非對角項後，L3 的 symmetric 權限從 100 降到 71 rad/s²（比值 0.42），因為要額外抵消 roll/pitch 耦合；對角近似會高估。\n4. yaw 比值恆為 1（見下方證明），但 yaw 權限本身隨 N 下降（single 307 → line4 19 rad/s²），這是 oracle 也逃不掉的物理上限。\n5. 所有推力相同的 hover 淨力矩恆為 0（含 L3、T4），所以 symmetric 制度可以 hover 不翻。\n\n" + PROOF % (d1, d2))
    open("results/p0/controllability.md", "w").write(md)
    print("\n".join(L)); print(PROOF % (d1, d2))
