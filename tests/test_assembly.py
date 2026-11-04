import numpy as np
from quadsim.params import CF2X, G, MODULE_PITCH
from quadsim.assembly import CONFIGS, assemble, sym_matrix

a, km = CF2X["a"], CF2X["km_ratio"]


def test_single_is_standard_x_mixer():
    # 標準 X 型：推力列全 1；τx = y·T；τy = −x·T；τz 轉向交錯
    A_std = np.array([[1, 1, 1, 1],
                      [-a, -a, a, a],
                      [-a, a, a, -a],
                      [km, -km, km, -km]])
    assert np.allclose(assemble(CONFIGS["single"])["A"], A_std)


def test_square4_com_at_geometric_center():
    assert np.allclose(assemble(CONFIGS["square4"])["com"], [MODULE_PITCH / 2, MODULE_PITCH / 2, 0])


def test_parallel_axis_line2():
    J1, J2 = assemble(CONFIGS["single"])["J"], assemble(CONFIGS["line2"])["J"]
    d = MODULE_PITCH / 2                       # 兩模組沿 x，各距 CoM d
    assert np.isclose(J2[1, 1], 2 * (CF2X["Iyy"] + CF2X["mass"] * d**2))
    assert J2[1, 1] > 3 * J1[1, 1]             # Iyy 明顯變大
    assert np.isclose(J2[0, 0], 2 * CF2X["Ixx"])   # Ixx 不變（沒有 y 偏移）


def test_square4_symmetric_hover_zero_torque():
    s = assemble(CONFIGS["square4"])
    u = np.full(4, s["M"] * G / (4 * s["N"]))   # 每個模組輸出同樣的 4 顆推力
    assert np.allclose(sym_matrix(s) @ u, [s["M"] * G, 0, 0, 0], atol=1e-12)


def test_equal_thrust_is_Mg_over_4N():
    for name, cells in CONFIGS.items():
        s = assemble(cells)
        T = np.full(4 * s["N"], s["M"] * G / (4 * s["N"]))
        assert np.isclose((s["A"] @ T)[0], s["M"] * G)
        assert np.isclose(T[0], CF2X["mass"] * G / 4)   # 每顆 = Mg/(4N) = 單機的 mg/4
        assert np.allclose((s["A"] @ T)[1:], 0, atol=1e-12)  # 全部相同 → 力矩為 0（CoM 是轉子重心）


def test_L3_inertia_offdiag_and_alpha():
    from analysis.controllability import lp_alpha
    s = assemble(CONFIGS["L3"]); J = s["J"]
    assert abs(J[0, 1]) > 0.1 * J[0, 0]                      # 有非對角項
    k = 0                                                     # 取 roll 的 +方向最佳解，驗證 α = J⁻¹τ 的另兩軸 = 0
    from scipy.optimize import linprog
    B = np.linalg.inv(J) @ s["A"][1:]
    r = linprog(-B[0], A_eq=np.vstack([s["A"][0], B[1:]]), b_eq=[s["M"] * G, 0, 0], bounds=(0, CF2X["Tmax"]), method="highs")
    alpha = np.linalg.inv(J) @ (s["A"] @ r.x)[1:]
    assert np.allclose(alpha[1:], 0, atol=1e-6) and np.isclose(alpha[0], lp_alpha(s["A"], s["M"], J, 0, +1))


def test_yaw_ratio_is_one():
    from analysis.controllability import table
    for r in table():
        assert np.isclose(r["ratio_yaw"], 1.0)


def test_khop_blind_equals_sym_and_distinct_equals_full():
    from analysis.controllability import table
    from analysis.khop import authority
    for r in table():
        s = assemble(CONFIGS[r["config"]])
        blind = authority(s, [0] * s["N"]); full = authority(s, list(range(s["N"])))
        assert np.allclose(blind, [r["alpha_sym_roll"], r["alpha_sym_pitch"], r["alpha_sym_yaw"], r["both_sym"]], rtol=1e-6)
        assert np.allclose(full, [r["alpha_full_roll"], r["alpha_full_pitch"], r["alpha_full_yaw"], r["both_full"]], rtol=1e-6)


def test_line_diameter_and_wl_refinement():
    from analysis.khop import neighbors, diameter_radius, refinements
    for n in (2, 3, 4, 6, 8):
        cells = [(i, 0) for i in range(n)]
        assert diameter_radius(neighbors(cells))[0] == n - 1
        steps = refinements(neighbors(cells))
        assert len(set(steps[-1][1])) == n          # 帶方向 → 左右兩端分得開，最終全部不同色


def test_cross_config_inf_exact_and_blind_shared():
    from analysis.cross_config import run, view_classes, fit, KS
    from quadsim.assembly import sym_matrix
    R = run(list(CONFIGS))
    assert max(r[1] for r in R["∞"].values()) < 1e-8          # 每個 (構型, 模組) 自成一類 → 可完全達成 G=I
    cls, V = view_classes("盲")
    assert V == 1                                              # 盲：全域只有一類，所有模組共用同一個 W
    W, _ = fit(list(CONFIGS), cls, V)
    for n, r in R["盲"].items():
        s = assemble(CONFIGS[n])
        assert np.allclose(r[0], np.linalg.inv(s["J"]) @ sym_matrix(s)[1:] @ W[0])


def test_per_axis_residual_bounded_or_flagged():
    from analysis.cross_config import run, run_struct, S3
    for R in (run(list(CONFIGS)), run(["single", "line2"]), run_struct(["single", "line2", "line3", "square4"])):
        for k in R.values():
            for r in k.values():
                assert np.isnan(r[2]) or r[2] <= S3 + 1e-9      # 適應後殘差不超過「沒控制力」基準，否則標為無法適應


def test_structural_infinity_ball_is_near_pseudoinverse_after_gain_adaptation():
    from analysis.cross_config import run_struct
    from quadsim.assembly import TRAIN
    S = run_struct(TRAIN); held = [n for n in CONFIGS if n not in TRAIN]
    inf, blind = S["∞"], S["盲"]
    assert np.mean([inf[n][1] for n in held]) < 0.5 * np.mean([blind[n][1] for n in held])   # 未適應也遠好於盲
    assert all(inf[n][2] < 0.3 for n in held if not np.isnan(inf[n][2]))                      # 逐軸增益適應後很小
