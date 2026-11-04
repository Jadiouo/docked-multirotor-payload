"""模組化對接多旋翼：單一模組的 CAD 與質量性質（CadQuery）。

用法（專案根目錄）：
    .venv/bin/python cad/module.py            # 產出 cad/out/ 底下全部檔案

座標系：原點 = 框架幾何中心，x/y 水平、z 向上，CAD 內部單位 mm，
輸出的 JSON 一律換成 SI（m、kg、kg·m^2）。

結構（照 ModQuad, ICRA 2018 的思路）：
  * 12 根碳纖維管，沿長方體 12 條邊
  * 8 個 ABS 角塊（薄殼立方體），管子插進角塊
  * 4 個側面各有 2 個 ABS 磁鐵座（clip），套在上、下邊的管子中點，
    座裡嵌 1 顆 D42-N52 磁鐵，磁鐵外表面和框架外表面（x 或 y = ±W/2）齊平
  * Crazyflie 2.1 只畫簡化外觀；質量、慣量用 lumped 值，不從幾何算
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cadquery as cq
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

# =============================================================================
# 參數（改這裡就會重新長出整個模組）
# =============================================================================
# --- 框架外形 ---------------------------------------------------------------
W = 116.0   # mm  框架水平邊長。出處：ModQuad ICRA 2018 "116x116x48 mm"
H = 48.0    # mm  框架高度。同上
# 模組中心距 pitch = W：相鄰模組的磁鐵面剛好貼在一起（磁鐵外表面在 ±W/2）。

# --- 碳纖維管 ----------------------------------------------------------------
TUBE_OD = 1.6   # mm  【假設】常見的 pultruded 碳管規格。ModQuad RAL 2019 提到桿徑約 1.37 mm
TUBE_ID = 0.8   # mm  【假設】
RHO_CF = 1.6e3  # kg/m^3  任務指定 1.6 g/cm^3

# --- ABS 角塊 ----------------------------------------------------------------
CORNER = 8.0    # mm  【假設】角塊立方體邊長。管子中心線離外表面 CORNER/2 = 4 mm
CORNER_WALL = 1.2   # mm  【假設】殼厚（內部中空，近似 3D 列印的薄殼 + 低填充）
TUBE_INSERT = 2.5   # mm  【假設】管子插入角塊的深度（從角塊內側面起算）
RHO_ABS = 1.05e3    # kg/m^3  任務指定 1.05 g/cm^3

# --- 磁鐵座 + 磁鐵 -----------------------------------------------------------
# 磁鐵放在側面中線（y=0 或 x=0）上、上下各一顆。選擇理由：
#   1. 兩顆磁鐵落在中線，左右對稱，法向吸力的合力作用點在面中心，不產生額外偏心力矩
#   2. 磁鐵座套在上下邊的管子中點，管子負責把力傳到角塊，不需要額外的豎桿
# 缺點：只有 2 顆磁鐵，面內扭轉（繞法線）抗力只靠磁鐵的剪力。
# 磁鐵有兩個選項（MAGNET 參數，預設 modquad_plate）：
#   "modquad_plate"：ModQuad 實機用的 NdFeB 方片 6.35 x 6.35 x 0.79 mm（docs/joint_design_notes.md 1.1）。
#       選它當預設的理由：results/p0/joint_loads.md 算出接頭最大負載約 2*mg（< 1 N），
#       D42 的強度遠超過需要，卻讓推重比掉到 1.46。
#   "d42"：K&J D42-N52 圓片 phi6.35 x 3.17 mm（docs/joint_design_notes.md 4.1），強但重。
# kind: 磁鐵外形；w: 直徑或邊長 mm；t: 厚度 mm。
MAGNETS = {
    "modquad_plate": dict(kind="square", w=6.35, t=0.79),
    "d42": dict(kind="round", w=6.35, t=3.17),
}
DEFAULT_MAGNET = "modquad_plate"
# 接頭強度上限 -> 見 joint_limits()
SHEAR_TO_PULL = 0.3    # 剪力/拉力，ModQuad RAL 2019 的磁鐵陣列實測（docs 1.3）；保守下限 0.15
RHO_MAG = 7.5e3     # kg/m^3  任務指定 NdFeB 7.5 g/cm^3（算出 0.75 g，和 K&J 標示 0.75 g 吻合）
CLIP_DEPTH = 5.0    # mm  【假設】磁鐵座沿法向的厚度，必須 > TUBE_OD/2 + 管子離表面距離
CLIP_WALL = 0.6     # mm  【假設】磁鐵座壁厚；磁鐵座另外兩個方向的邊長 = 磁鐵寬 + 2*壁厚（兩種磁鐵都是 7.55 mm）
# 管子中心線離外表面 = CORNER/2，所以兩顆磁鐵的間距由幾何決定：
MAG_SPACING = H - CORNER    # mm  上下磁鐵中心距 d = 40 mm

# --- Crazyflie 2.1 -----------------------------------------------------------
CF_MASS = 0.029   # kg  Bitcraze datasheet "Takeoff weight: 29g"
CF_MOTOR_OFFSET = 28.0   # mm  馬達 x、y 偏移，quadsim/params.py 的 a = 0.028 m
CF_PROP_R = 23.0         # mm  螺旋槳半徑 0.023 m（params.py 註解）
# cf2x 慣量（quadsim/params.py，來自 cf2x.urdf，質量 0.027 kg）：
CF2X_MASS = 0.027
CF2X_INERTIA_DIAG = (1.4e-5, 1.4e-5, 2.17e-5)
# 29 g 的整機慣量 = cf2x 慣量 × (29/27)。【假設】質量分佈形狀和 cf2x 相同，只是整體變重
CF_INERTIA_DIAG = tuple(i * CF_MASS / CF2X_MASS for i in CF2X_INERTIA_DIAG)
CF_COM = (0.0, 0.0, 0.0)  # mm  【假設】Crazyflie 質心在框架幾何中心（PCB 中心）
# --- 以下只影響外觀（不進質量計算） ---
CF_PCB = (30.0, 30.0, 1.0)    # mm  中央 PCB 方板 x, y, 厚
CF_ARM_W = 4.0                # mm  支臂寬
CF_MOTOR_D, CF_MOTOR_L = 7.0, 15.0   # mm  馬達直徑、長度（頂面 = z +7）
CF_MOTOR_TOP = 7.0
CF_PROP_T = 0.5               # mm  螺旋槳薄圓盤厚
CF_BATT = (30.0, 13.0, 7.0)   # mm  電池，貼在 PCB 下方

# --- 輸出 ---------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "out"
MM = 1e-3   # mm -> m


# =============================================================================
# 幾何：小工具
# =============================================================================
def _cyl(radius, length, center, axis):
    """以 center 為中心、沿 axis（單位向量）的圓柱。"""
    ax = cq.Vector(*axis)
    start = cq.Vector(*center) - ax * (length / 2)
    return cq.Solid.makeCylinder(radius, length, start, ax)


def _box(sx, sy, sz, center):
    c = cq.Vector(*center)
    return cq.Solid.makeBox(sx, sy, sz, c - cq.Vector(sx / 2, sy / 2, sz / 2))


def _sgn():
    return (-1, 1)


# =============================================================================
# 幾何：框架
# =============================================================================
def _node(sx, sy, sz):
    """管子中心線的交點（角塊內的「節點」）。離外表面 CORNER/2。"""
    return (sx * (W / 2 - CORNER / 2), sy * (W / 2 - CORNER / 2), sz * (H / 2 - CORNER / 2))


def make_tubes():
    """12 根碳管。每根從一個角塊內插入、到另一個角塊內，長度 = 邊長 - 2*(CORNER - TUBE_INSERT)。"""
    tubes = []
    hole = 2 * (CORNER - TUBE_INSERT)
    for axis, outer in (((1, 0, 0), W), ((0, 1, 0), W), ((0, 0, 1), H)):
        length = outer - hole
        ia = axis.index(1)
        others = [i for i in range(3) if i != ia]
        full = (W, W, H)
        for s1 in _sgn():
            for s2 in _sgn():
                center = [0.0, 0.0, 0.0]
                center[others[0]] = s1 * (full[others[0]] / 2 - CORNER / 2)
                center[others[1]] = s2 * (full[others[1]] / 2 - CORNER / 2)
                outer_cyl = _cyl(TUBE_OD / 2, length, center, axis)
                inner_cyl = _cyl(TUBE_ID / 2, length + 1, center, axis)
                tubes.append(outer_cyl.cut(inner_cyl))
    return tubes


def make_corners():
    """8 個 ABS 角塊：立方體 - 內部空腔 - 三個管子孔（只開在朝內的三個面）。"""
    corners = []
    cav = CORNER - 2 * CORNER_WALL
    for sx in _sgn():
        for sy in _sgn():
            for sz in _sgn():
                c = _node(sx, sy, sz)
                solid = _box(CORNER, CORNER, CORNER, c).cut(_box(cav, cav, cav, c))
                # 朝內的面中心：節點往內退 CORNER/2
                for i, s in enumerate((sx, sy, sz)):
                    axis = [0, 0, 0]
                    axis[i] = 1
                    face = list(c)
                    face[i] = c[i] - s * CORNER / 2
                    # 孔從內側面往外切，深度 = CORNER_WALL（穿過殼壁進入空腔）
                    hole_c = list(face)
                    hole_c[i] = face[i] + s * CORNER_WALL / 2
                    solid = solid.cut(_cyl(TUBE_OD / 2, CORNER_WALL + 0.0, hole_c, axis))
                corners.append(solid)
    return corners


def _rot_z(shape, k):
    """繞 z 軸轉 k*90 度。"""
    return shape.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), 90 * k)


def _rot_vec_z(v, k):
    """向量繞 z 軸轉 k*90 度（整數運算，避免浮點誤差）。"""
    x, y, z = v
    for _ in range(k % 4):
        x, y = -y, x
    return (x, y, z)


# 四個面：(名稱, 繞 z 轉幾個 90 度)。+x 面是基準，其他面由它轉出來。
FACES = (("+x", 0), ("+y", 1), ("-x", 2), ("-y", 3))


def _clip_size(mag):
    return mag["w"] + 2 * CLIP_WALL


def _mag_solid(mag, center):
    """磁鐵實體，軸（厚度方向）沿 x。"""
    if mag["kind"] == "round":
        return _cyl(mag["w"] / 2, mag["t"], center, (1, 0, 0))
    return _box(mag["t"], mag["w"], mag["w"], center)


def make_clips_and_magnets(magnet=DEFAULT_MAGNET):
    """每面 2 組（上、下）磁鐵座 + 磁鐵。先在 +x 面做，再繞 z 轉到其他面。"""
    mag = MAGNETS[magnet]
    CLIP_SIZE = _clip_size(mag)
    clips, magnets, mag_info = [], [], []
    node_off = CORNER / 2                    # 管子中心線離外表面
    for fname, k in FACES:
        for sz in _sgn():
            z = sz * (H / 2 - CORNER / 2)
            # 磁鐵座：x 方向從外表面往內 CLIP_DEPTH，管子沿 y 穿過
            cx = W / 2 - CLIP_DEPTH / 2
            clip = _box(CLIP_DEPTH, CLIP_SIZE, CLIP_SIZE, (cx, 0, z))
            clip = clip.cut(_cyl(TUBE_OD / 2, CLIP_SIZE + 1, (W / 2 - node_off, 0, z), (0, 1, 0)))
            pocket_c = (W / 2 - mag["t"] / 2, 0, z)
            clip = clip.cut(_mag_solid(mag, pocket_c))
            magnet = _mag_solid(mag, pocket_c)
            clips.append(_rot_z(clip, k))
            magnets.append(_rot_z(magnet, k))

            c = _rot_vec_z(pocket_c, k)
            # 極性（只記錄，不建模）：
            #   +x、+y 面：上方磁鐵朝外 N，下方朝外 S；
            #   -x、-y 面：上方朝外 S，下方朝外 N。
            #   A 的 +x 面貼 B 的 -x 面時，同高度的一對朝外極性相反 -> 相吸。
            #   代價：模組不能轉 180 度（yaw 必須一致），這和編隊飛行時 yaw 對齊的假設一致。
            plus = fname in ("+x", "+y")
            top_pole = "N" if plus else "S"
            pole = top_pole if sz > 0 else ("S" if top_pole == "N" else "N")
            normal = [round(v) for v in _rot_vec_z((1, 0, 0), k)]
            mag_info.append(dict(face=fname, level="top" if sz > 0 else "bottom",
                                 center_m=[round(v * MM, 9) + 0.0 for v in c],
                                 outward_normal=normal, outward_pole=pole))
    return clips, magnets, mag_info


# =============================================================================
# 幾何：Crazyflie 2.1（簡化外觀）
# =============================================================================
def make_crazyflie():
    """回傳 dict：pcb、arms、motors、battery、props（都只是外觀）。"""
    a = CF_MOTOR_OFFSET
    pcb = _box(*CF_PCB, (0, 0, 0))
    arms, motors, props = [], [], []
    for sx in _sgn():
        for sy in _sgn():
            length = math.hypot(a, a)
            ang = math.degrees(math.atan2(sy * a, sx * a))
            arm = _box(length, CF_ARM_W, CF_PCB[2], (length / 2, 0, 0)).rotate(
                cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), ang)
            arms.append(arm)
            mz = CF_MOTOR_TOP - CF_MOTOR_L / 2
            motors.append(_cyl(CF_MOTOR_D / 2, CF_MOTOR_L, (sx * a, sy * a, mz), (0, 0, 1)))
            props.append(_cyl(CF_PROP_R, CF_PROP_T, (sx * a, sy * a, CF_MOTOR_TOP + CF_PROP_T / 2), (0, 0, 1)))
    battery = _box(*CF_BATT, (0, 0, -CF_PCB[2] / 2 - CF_BATT[2] / 2))
    return dict(pcb=pcb, arms=arms, motors=motors, battery=battery, props=props)


# =============================================================================
# 質量性質
# =============================================================================
def _solid_props(shape):
    """單位密度下的 (體積 mm^3, COM mm, 對自身 COM 的慣性矩陣 mm^5)。"""
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.wrapped, p)
    m = p.MatrixOfInertia()   # OCC 的慣性矩陣是對質心算的（已用偏移的方塊驗證）
    I = [[m.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)]
    c = p.CentreOfMass()
    return p.Mass(), (c.X(), c.Y(), c.Z()), I


def _matmul_shift(m, r):
    """平行軸：質量 m、位移 r (m) 的點質量慣量 m(|r|^2 E - r r^T)。"""
    r2 = sum(x * x for x in r)
    return [[m * ((r2 if i == j else 0.0) - r[i] * r[j]) for j in range(3)] for i in range(3)]


def _madd(a, b):
    return [[a[i][j] + b[i][j] for j in range(3)] for i in range(3)]


def _group(shapes, rho):
    """一群同材料零件 -> 質量 kg、質心 m（對模組原點）、對模組原點的慣量 kg m^2。"""
    mass = 0.0
    mom = [0.0, 0.0, 0.0]            # 質量 × 質心
    I_origin = [[0.0] * 3 for _ in range(3)]
    for s in shapes:
        vol, com, I = _solid_props(s)
        m = vol * (MM ** 3) * rho
        c = [x * MM for x in com]
        Ic = [[I[i][j] * (MM ** 5) * rho for j in range(3)] for i in range(3)]
        mass += m
        for i in range(3):
            mom[i] += m * c[i]
        I_origin = _madd(I_origin, _madd(Ic, _matmul_shift(m, c)))
    com = [x / mass for x in mom]
    return mass, com, I_origin


def _to_com(mass, com, I_origin):
    return _madd(I_origin, [[-v for v in row] for row in _matmul_shift(mass, com)])


def _round(x, sig=10):
    if isinstance(x, (list, tuple)):
        return [_round(v, sig) for v in x]
    if abs(x) < 1e-15:      # 數值雜訊（對稱性應為 0 的項）一律歸零，避免重跑時符號翻轉
        return 0.0
    return float(f"{x:.{sig}g}")


def build_parts(magnet=DEFAULT_MAGNET):
    tubes = make_tubes()
    corners = make_corners()
    clips, magnets, mag_info = make_clips_and_magnets(magnet)
    cf = make_crazyflie()
    return dict(magnet=magnet, tubes=tubes, corners=corners, clips=clips, magnets=magnets,
                mag_info=mag_info, cf=cf)


def prop_clearance(parts):
    """螺旋槳圓盤到框架（管子 + 角塊 + 磁鐵座 + 磁鐵）的最短 3D 距離 (mm)。"""
    frame = cq.Compound.makeCompound(parts["tubes"] + parts["corners"] + parts["clips"] + parts["magnets"])
    best = 1e9
    for pr in parts["cf"]["props"]:
        d = BRepExtrema_DistShapeShape(pr.wrapped, frame.wrapped)
        d.Perform()
        best = min(best, d.Value())
    return best


def joint_limits(magnet, spacing_mm=MAG_SPACING):
    """每個接頭（一個側面 = 2 對磁鐵）的強度上限。全部是推論，沒有實測。

    modquad_plate：ModQuad ICRA 2018 "面對面 4 顆磁鐵等效 1 kg"（docs 1.1）。
        我們的接頭面對面有 2 顆 + 2 顆 = 4 顆，所以拉力上限 = 1 kgf = 9.81 N。
        （若論文的「4 顆」是指每側 4 顆，上限會是 2 倍；取較保守的解讀。）
    d42：單顆吸鋼板 1.29 kgf（K&J），磁鐵對磁鐵取 50%（K&J 部落格），2 顆：
        1.29 * 9.81 * 0.5 * 2 = 12.65 N（docs 第 6 節 "約 12.7 N"）。
    剪力 = 拉力 * 0.3（ModQuad RAL 2019 磁鐵陣列的剪力/拉力）。
    剝離力矩 = 拉力 * d/2：以一側磁鐵為支點，另一顆被拉開的力臂約為 d/2 的粗估（推論）。
    """
    g = 9.81
    pull = {"modquad_plate": 1.0 * g, "d42": 1.29 * g * 0.5 * 2}[magnet]
    basis = {
        "modquad_plate": "ModQuad ICRA 2018: 面對面 4 顆等效 1 kg -> 1 kgf = 9.81 N / 接頭（推論）",
        "d42": "K&J D42-N52 吸鋼板 1.29 kgf x 50%（磁對磁）x 2 顆 = 12.65 N（推論）",
    }[magnet]
    return dict(
        pull_N=_round(pull),
        shear_N=_round(pull * SHEAR_TO_PULL),
        peel_moment_Nm=_round(pull * spacing_mm * MM / 2),
        shear_to_pull=SHEAR_TO_PULL,
        sources=dict(
            pull_N="推論：" + basis,
            shear_N="推論：拉力 x 0.3（比值出處 ModQuad RAL 2019，docs 1.3；保守下限 0.15）",
            peel_moment_Nm="推論：拉力 x d/2，d = 磁鐵上下間距（粗估，需實測）",
        ),
    )


def compute_props(parts=None):
    parts = parts or build_parts()
    mag = MAGNETS[parts["magnet"]]
    groups = {
        "carbon_tubes": _group(parts["tubes"], RHO_CF),
        "abs_corners": _group(parts["corners"], RHO_ABS),
        "abs_magnet_clips": _group(parts["clips"], RHO_ABS),
        "magnets_ndfeb": _group(parts["magnets"], RHO_MAG),
    }
    cf_com = [x * MM for x in CF_COM]
    cf_I_com = [[CF_INERTIA_DIAG[i] if i == j else 0.0 for j in range(3)] for i in range(3)]
    cf_I_origin = _madd(cf_I_com, _matmul_shift(CF_MASS, cf_com))

    mass = CF_MASS + sum(g[0] for g in groups.values())
    mom = [CF_MASS * cf_com[i] + sum(g[0] * g[1][i] for g in groups.values()) for i in range(3)]
    com = [x / mass for x in mom]
    I_origin = cf_I_origin
    for g in groups.values():
        I_origin = _madd(I_origin, g[2])
    I_com = _to_com(mass, com, I_origin)

    frame_mass = sum(groups[k][0] for k in ("carbon_tubes", "abs_corners", "abs_magnet_clips"))
    clearance = prop_clearance(parts) * MM
    return dict(
        units=dict(length="m", mass="kg", inertia="kg*m^2", note="座標原點 = 框架幾何中心；z 向上"),
        mass=_round(mass),
        com=_round(com),
        inertia=_round(I_com),
        pitch=_round(W * MM),
        pitch_note="相鄰模組中心距 = 框架外形 W（磁鐵外表面在 +-W/2，面對面貼合）",
        frame_size_m=_round([W * MM, W * MM, H * MM]),
        magnet_spacing_m=_round(MAG_SPACING * MM),
        magnet_type=parts["magnet"],
        magnet_size_mm=dict(kind=mag["kind"], width=mag["w"], thickness=mag["t"]),
        joint_limits=joint_limits(parts["magnet"]),
        mass_breakdown_kg=dict(
            crazyflie_2_1=_round(CF_MASS),
            carbon_tubes=_round(groups["carbon_tubes"][0]),
            abs_corners=_round(groups["abs_corners"][0]),
            abs_magnet_clips=_round(groups["abs_magnet_clips"][0]),
            magnets_ndfeb=_round(groups["magnets_ndfeb"][0]),
            frame_without_magnets=_round(frame_mass),
        ),
        magnet_mass_each_kg=_round(groups["magnets_ndfeb"][0] / len(parts["magnets"])),
        magnets=parts["mag_info"],
        crazyflie_lumped=dict(mass=CF_MASS, inertia_diag=_round(list(CF_INERTIA_DIAG)),
                              com_m=_round(cf_com)),
        prop_clearance_m=_round(clearance),
        prop_radius_m=CF_PROP_R * MM,
        sources={
            "mass, com, inertia": "CAD計算（CadQuery 實體體積 x 密度；Crazyflie 為 lumped，兩者平行軸合成）",
            "pitch": "CAD計算（= 框架外形 W；W=116 mm 為 ModQuad ICRA 2018 規格）",
            "frame_size_m": "規格書（ModQuad ICRA 2018: 116x116x48 mm）",
            "crazyflie_lumped.mass": "規格書（Bitcraze datasheet: takeoff weight 29 g）",
            "crazyflie_lumped.inertia_diag": "假設（quadsim/params.py cf2x 慣量 x 29/27；cf2x 值來自 cf2x.urdf）",
            "crazyflie_lumped.com_m": "假設（質心在框架幾何中心）",
            "carbon_tubes": "CAD計算；OD 1.6 / ID 0.8 mm 為假設，密度 1.6 g/cm3 為任務指定",
            "abs_corners": "CAD計算；邊長 8 mm、殼厚 1.2 mm 為假設，密度 1.05 g/cm3 為任務指定",
            "joint_limits": "推論（見 joint_limits 內的 sources；沒有實測）",
            "abs_magnet_clips": "CAD計算；尺寸為假設，密度 1.05 g/cm3 為任務指定",
            "magnets_ndfeb": "CAD計算（尺寸：modquad_plate 為論文規格 6.35x6.35x0.79 mm，d42 為 K&J 規格書 phi6.35x3.17 mm；密度 7.5 g/cm3 為任務指定）",
            "magnets (positions)": "CAD計算（磁鐵位置為設計選擇：側面中線、上下各一）",
            "magnet_spacing_m": "CAD計算（= H - CORNER，由管子位置決定）",
            "prop_clearance_m": "CAD計算（槳盤 r=23 mm 到框架實體的最短 3D 距離；槳半徑來自 params.py 註解）",
        },
    )


# =============================================================================
# 輸出
# =============================================================================
COLORS = dict(
    tubes=cq.Color(0.1, 0.1, 0.1), corners=cq.Color(0.95, 0.55, 0.1),
    clips=cq.Color(0.95, 0.55, 0.1), magnets=cq.Color(0.7, 0.7, 0.75),
    pcb=cq.Color(0.1, 0.5, 0.2), arms=cq.Color(0.1, 0.5, 0.2),
    motors=cq.Color(0.4, 0.4, 0.45), battery=cq.Color(0.2, 0.3, 0.7),
    props=cq.Color(0.5, 0.7, 0.95, 0.5),
)


def make_assembly(parts, name="module"):
    asm = cq.Assembly(name=name)
    cf = parts["cf"]
    groups = {"tubes": parts["tubes"], "corners": parts["corners"], "clips": parts["clips"],
              "magnets": parts["magnets"], "pcb": [cf["pcb"]], "arms": cf["arms"],
              "motors": cf["motors"], "battery": [cf["battery"]], "props": cf["props"]}
    for key, shapes in groups.items():
        for i, s in enumerate(shapes):
            asm.add(cq.Workplane(obj=s), name=f"{key}_{i}", color=COLORS[key])
    return asm


def all_shapes(parts):
    cf = parts["cf"]
    return (parts["tubes"] + parts["corners"] + parts["clips"] + parts["magnets"]
            + [cf["pcb"], cf["battery"]] + cf["arms"] + cf["motors"] + cf["props"])


def export_all(out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # D42 版本只輸出 JSON（幾何檔以預設版本為準）
    d42 = compute_props(build_parts("d42"))
    (out_dir / "module_props_d42.json").write_text(json.dumps(d42, indent=2, ensure_ascii=False) + "\n")
    parts = build_parts()
    props = compute_props(parts)
    (out_dir / "module_props.json").write_text(json.dumps(props, indent=2, ensure_ascii=False) + "\n")

    asm = make_assembly(parts)
    asm.export(str(out_dir / "module.step"))
    compound = cq.Workplane(obj=cq.Compound.makeCompound(all_shapes(parts)))
    cq.exporters.export(compound, str(out_dir / "module.stl"), tolerance=0.02, angularTolerance=0.2)

    # SVG 投影：CadQuery 對非軸向的投影方向會把畫面轉歪，所以側視圖的做法是
    # 先把實體繞 x 軸轉 -90 度（相機從 -y 看過去 = 從 +z 往下看），再用俯視投影
    side = compound.rotate((0, 0, 0), (1, 0, 0), -90)
    views = {"top": (compound, (0, 0, 1)), "side": (side, (0, 0, 1)), "iso": (compound, (1, -1, 0.8))}
    for name, (wp, d) in views.items():
        cq.exporters.export(wp, str(out_dir / f"module_{name}.svg"), opt={
            "projectionDir": d, "width": 700, "height": 700, "showAxes": False,
            "strokeWidth": 0.15, "showHidden": False, "marginLeft": 20, "marginTop": 20})

    # 3 個模組沿 x 對接，間距 = W
    line = cq.Assembly(name="line3")
    for i in (-1, 0, 1):
        line.add(make_assembly(parts, name=f"module{i + 1}"), loc=cq.Location(cq.Vector(i * W, 0, 0)))
    line.export(str(out_dir / "line3.step"))
    return props


if __name__ == "__main__":
    p = export_all()
    print(json.dumps({k: p[k] for k in ("mass", "com", "inertia", "pitch", "mass_breakdown_kg",
                                        "prop_clearance_m")}, indent=2))
