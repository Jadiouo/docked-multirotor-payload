"""MuJoCo 參考模型：整架 = 單一 free body，用來驗證 quadsim/env.py 的積分器，也可用來渲染。

- body 原點 = 整架 CoM（所以 qpos[:3] 就是 CoM 位置，xfrc_applied 的力矩也是對 CoM）。
- 質量、慣量直接指定成 assemble() 算出的 M、J（fullinertia），不讓 MuJoCo 從外觀幾何自己算。
- 外觀：每個模組放一份 cad/out/module.stl（單位 mm，scale 0.001），不參與碰撞。
- 推力與 yaw 反扭矩每個子步用 xfrc_applied 施加：它是「世界座標」的力與力矩，作用在 body 的 CoM，
  所以機體系的 F、τ 要先乘 R（body→world）。
- 關掉接觸，只留重力；積分器 Euler、dt = 0.002，和 env 的半隱式 Euler 同一種離散化。
  （MuJoCo 的 Euler：先 qvel += dt·qacc，再用新的 qvel 積分 qpos，四元數乘 exp(ω dt)；free joint 的 ω 在機體座標。）
"""
from pathlib import Path
import mujoco
import numpy as np
from .params import G

STL = Path(__file__).resolve().parent.parent / "cad" / "out" / "module.stl"


def build_mjcf(asm, centers, dt=0.002, stl=STL, inertial_pos=(0., 0., 0.)):
    """asm 需有 M、J（J 為對 CoM 的慣量）；centers = 各模組中心相對 body 原點的位置 (N,3)。回傳 MJCF 字串。
    inertial_pos = CoM 相對 body 原點；預設 0（原點即 CoM）。獨立檢查力模型時把原點放在幾何原點、這裡填 CoM。"""
    J = asm["J"]
    full = " ".join(f"{x:.12g}" for x in (J[0, 0], J[1, 1], J[2, 2], J[0, 1], J[0, 2], J[1, 2]))  # M00 M11 M22 M01 M02 M12
    geoms = "\n".join(f'      <geom type="mesh" mesh="module" pos="{c[0]:.9g} {c[1]:.9g} {c[2]:.9g}" '
                      f'contype="0" conaffinity="0" rgba="0.6 0.65 0.7 1"/>' for c in centers)
    return f"""<mujoco model="modular_multirotor">
  <compiler angle="radian"/>
  <option timestep="{dt}" integrator="Euler" gravity="0 0 -{G}">
    <flag contact="disable"/>
  </option>
  <asset>
    <mesh name="module" file="{Path(stl).resolve()}" scale="0.001 0.001 0.001"/>
  </asset>
  <worldbody>
    <body name="craft">
      <freejoint name="root"/>
      <inertial pos="{inertial_pos[0]:.12g} {inertial_pos[1]:.12g} {inertial_pos[2]:.12g}" mass="{asm['M']:.12g}" fullinertia="{full}"/>
{geoms}
    </body>
  </worldbody>
</mujoco>"""


def make_reference(asm, centers, dt=0.002, inertial_pos=(0., 0., 0.)):
    model = mujoco.MjModel.from_xml_string(build_mjcf(asm, centers, dt, inertial_pos=inertial_pos))
    return model, mujoco.MjData(model)


def set_state(model, data, pos, vel, quat, omega_body):
    """qpos = [p(3), q(w,x,y,z)]；qvel = [v 世界座標(3), ω 機體座標(3)]。"""
    data.qpos[:3], data.qpos[3:7] = pos, quat
    data.qvel[:3], data.qvel[3:6] = vel, omega_body
    mujoco.mj_forward(model, data)


def step(model, data, asm, thrust):
    """施加 4N 顆推力 thrust（N）並前進一個 dt。F = [0,0,ΣT]、τ = A[1:]·T（機體系）→ 乘 R 變世界座標。"""
    R = data.xmat[1].reshape(3, 3)                     # body→world；先 forward 過，xmat 對應目前的 qpos
    F_b = np.array([0., 0., thrust.sum()])
    tau_b = asm["A"][1:] @ thrust
    data.xfrc_applied[1, :3] = R @ F_b
    data.xfrc_applied[1, 3:] = R @ tau_b
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)                     # 讓 xmat 跟上新的 qpos，下一步施力用得到


def step_per_rotor(model, data, rotor_pos_body, spin_torque_body, thrust):
    """獨立的力模型：不自己算 r×F。每顆轉子的世界推力 R·(T_j ẑ) 用 mj_applyFT 施加在轉子的世界座標位置
    （body 位姿 + R·轉子在 body 座標的位置，body 原點 = 幾何原點），MuJoCo 自己換算成對 CoM 的力矩。
    yaw 反扭矩是唯一要自己算的項：spin_torque_body = 每單位推力的反扭矩 km·spin_j，沿 body ẑ。"""
    R = data.xmat[1].reshape(3, 3)
    data.qfrc_applied[:] = 0
    zero = np.zeros(3)
    for T, r, k in zip(thrust, rotor_pos_body, spin_torque_body):
        mujoco.mj_applyFT(model, data, R @ np.array([0., 0., T]), R @ np.array([0., 0., k * T]),
                          data.xpos[1] + R @ r, 1, data.qfrc_applied)
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
