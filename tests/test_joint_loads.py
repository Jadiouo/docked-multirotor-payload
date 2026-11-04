import numpy as np
import pytest
from quadsim.params import CF2X, G, MODULE_PITCH
from quadsim.assembly import CONFIGS, assemble, MOTOR_XY
from analysis.joint_loads import (build, module_loads, joints, project, allocate, alloc_oracle, alloc_khop, alpha_max,
                                  rigid_body, edges, is_tree, wrench, METHODS, KEYS)

m, a, p = CF2X["mass"], CF2X["a"], MODULE_PITCH
TREES = [n for n in CONFIGS if is_tree(build(CONFIGS[n]))]
LOOPS = ["square4", "grid6", "grid9", "grid10"]


def test_build_matches_assemble():
    for n, c in CONFIGS.items():
        b, s = build(c), assemble(c)
        assert np.allclose(b["J"], s["J"]) and np.allclose(b["A"], s["A"]) and np.allclose(b["com"], s["com"])


def test_load_sums_zero_all_cases():
    # module_loads 內部 assert 總和為 0；含 clip、非零 ω、payload
    rng = np.random.default_rng(0)
    for n in ["line3", "T4", "grid6", "plus5"]:
        mod = build(CONFIGS[n])
        for meth in METHODS:
            T = allocate(mod, meth, np.array([30., -20., 10.]), omega=np.array([3., -2., 5.]))
            module_loads(mod, T, omega=np.array([3., -2., 5.]))
        module_loads(mod, rng.uniform(0, CF2X["Tmax"], 4 * mod["N"]), omega=rng.normal(size=3))
    pm = build(CONFIGS["line3"], payload=0.01, host=2)
    module_loads(pm, alloc_oracle(pm, np.zeros(3)))


def test_line2_hover_zero_joint_force():
    mod = build(CONFIGS["line2"]); T = alloc_oracle(mod, np.zeros(3))
    assert np.allclose(T, m * G / 4)                      # 每顆 mg/4：每個模組自己扛自己的重量
    W = joints(mod, module_loads(mod, T))
    assert np.abs(np.array(list(W.values()))).max() < 1e-12


def test_line3_oracle_pitch_hand_calc_section_method():
    """手算：葉端模組 0（d=(−p,0,0)）。α=(0,α,0)，oracle：ΔT_k = −x_k·Jyy·α/Σx_k²（line 上 AAᵀ 為對角）。
    模組 0 外載垂直力 E = 4pc − m·α·p（c = Jyy α/Σx²），繞自己中心的 y 力矩 M0 = 4a²c − Iyy·α。
    接頭 (0,1) 在模組 0 中心 +x 方向 p/2 處施力 −E：m_y = −M0 − pE/2。"""
    mod = build(CONFIGS["line3"]); al = 40.0
    T = alloc_oracle(mod, np.array([0, al, 0.])); assert T.min() > 0 and T.max() < CF2X["Tmax"]
    x = mod["pos"][:, 0]; Jyy = mod["J"][1, 1]
    c = Jyy * al / (x**2).sum()
    E = 4 * p * c - m * al * p
    M0 = 4 * a**2 * c - CF2X["Iyy"] * al
    my = -M0 - p * E / 2
    W = joints(mod, module_loads(mod, T), "tree")
    f, mm = W[(0, 1)][:3], W[(0, 1)][3:]
    # 接頭對 j=1 的力 = +E（對 0 的力 = −E）
    assert np.isclose(f[2], E, rtol=1e-9) and np.allclose(f[:2], 0, atol=1e-12)
    # 對 1 的力矩：以中點為參考，對 0 的 m_y 的反作用
    assert np.isclose(mm[1], -my, rtol=1e-9)
    rows = {r["edge"]: r for r in project(mod, W)}
    assert np.isclose(rows[(0, 1)]["shear_oop"], abs(E), rtol=1e-9) and np.isclose(rows[(0, 1)]["bend"], abs(my), rtol=1e-9)
    assert abs(E) > 1e-6        # 數值不是平凡的 0


def test_statically_determinate_independent_of_stiffness():
    for n in TREES:
        if n == "single": continue
        mod = build(CONFIGS[n])
        for meth in ("oracle", "sym", "khop0"):
            ext = module_loads(mod, allocate(mod, meth, np.array([40., 30., 10.])))
            ref = joints(mod, ext, "tree")
            for kt, kr in [(1e4, 25.), (1e2, 25.), (1e4, 250.), (1e6, 0.5)]:
                fr = joints(mod, ext, "frame", kt, kr)
                assert all(np.allclose(ref[e], fr[e], atol=1e-9) for e in ref), (n, meth, kt, kr)


def test_loop_depends_on_stiffness_and_equilibrium():
    mod = build(CONFIGS["grid6"]); ext = module_loads(mod, allocate(mod, "oracle", np.array([30., 40., 20.])))
    w1, w2 = joints(mod, ext, "frame", 1e4, 25.), joints(mod, ext, "frame", 1e4, 250.)
    assert max(np.abs(w1[e] - w2[e]).max() for e in w1) > 1e-4


def test_khop_with_whole_ball_equals_oracle():
    mod = build(CONFIGS["line3"]); al = np.array([0, 40., 20.])
    assert np.allclose(alloc_khop(mod, al, 5), alloc_oracle(mod, al))


def test_yaw_has_inplane_shear_all_methods():
    for n in ["line2", "line3", "square4"]:
        mod = build(CONFIGS[n]); am = alpha_max(mod, 2)
        for meth in METHODS:
            T = allocate(mod, meth, np.array([0, 0, 0.5 * am]))
            r = project(mod, joints(mod, module_loads(mod, T)))
            assert max(x["shear_ip"] for x in r) > 1e-5
