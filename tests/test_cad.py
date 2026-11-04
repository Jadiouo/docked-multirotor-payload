"""CAD 模組的驗收測試。需要 cadquery 與 mujoco（都在 .venv）。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

cq = pytest.importorskip("cadquery")
mujoco = pytest.importorskip("mujoco")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cad"))
import module as M  # noqa: E402

OUT = ROOT / "cad" / "out"


@pytest.fixture(scope="module")
def parts():
    return M.build_parts()


@pytest.fixture(scope="module")
def props(parts):
    return M.compute_props(parts)


def test_deterministic_and_matches_json(props):
    again = M.compute_props(M.build_parts())
    assert again == props                                   # 同一程序內重跑
    saved = json.loads((OUT / "module_props.json").read_text())
    assert saved["mass"] == pytest.approx(props["mass"], rel=1e-9)
    assert np.allclose(saved["inertia"], props["inertia"], rtol=1e-9, atol=1e-18)
    assert np.allclose(saved["com"], props["com"], atol=1e-12)


def test_total_mass_in_modquad_range(props):
    assert 0.035 <= props["mass"] <= 0.045                  # ModQuad 實機 36.7–40 g
    assert 0.005 <= props["mass_breakdown_kg"]["frame_without_magnets"] <= 0.009   # ModQuad 框架 7 g


def test_inertia_symmetric_positive_definite(props):
    I = np.array(props["inertia"])
    assert np.allclose(I, I.T, atol=1e-15)
    assert np.all(np.linalg.eigvalsh(I) > 0)


def test_prop_clears_frame(props):
    assert props["prop_clearance_m"] > 0


def test_magnets_on_faces_and_pitch(props):
    assert props["pitch"] == pytest.approx(0.116)
    mags = props["magnets"]
    assert len(mags) == 8
    for m in mags:
        c, n = np.array(m["center_m"]), np.array(m["outward_normal"])
        # 磁鐵中心在外表面內 MAG_T/2 處
        assert c @ n == pytest.approx(M.W / 2 * M.MM - M.MAGNETS[M.DEFAULT_MAGNET]['t'] / 2 * M.MM, abs=1e-9)


def test_line3_modules_touch_without_overlap(parts):
    """相鄰模組只在磁鐵面貼合，實體不重疊。"""
    shapes = M.all_shapes(parts)
    a = cq.Compound.makeCompound(shapes)
    b = a.moved(cq.Location(cq.Vector(M.W, 0, 0)))
    assert a.intersect(b).Volume() < 1e-6                   # mm^3


def test_stl_loads_in_mujoco():
    xml = f"""<mujoco><asset><mesh name="m" file="{OUT / 'module.stl'}" scale="0.001 0.001 0.001"/></asset>
    <worldbody><body><freejoint/><geom type="mesh" mesh="m"/></body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nmesh == 1 and model.nmeshvert > 0


def test_d42_option_and_joint_limits(props):
    d42 = json.loads((OUT / "module_props_d42.json").read_text())
    assert d42["magnet_type"] == "d42" and props["magnet_type"] == "modquad_plate"
    assert d42["mass"] > props["mass"]
    assert props["joint_limits"]["pull_N"] == pytest.approx(9.81)
    assert props["joint_limits"]["shear_N"] == pytest.approx(0.3 * 9.81)
    assert d42["joint_limits"]["shear_N"] == pytest.approx(3.8, abs=0.05)
    again = M.compute_props(M.build_parts("d42"))
    assert again["mass"] == pytest.approx(d42["mass"], rel=1e-9)
    assert again["prop_clearance_m"] > 0
