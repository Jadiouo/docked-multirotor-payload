"""模組化四旋翼的組裝：整架的質量、質心、慣量與 allocation matrix。
座標：x 前、y 左、z 上（機體系，所有模組朝向相同）。力矩 = r × F，推力沿 +z。"""
import numpy as np
from .params import CF2X, G, MODULE_PITCH

# 格點構型（整數格點 × MODULE_PITCH = 模組中心位置）
CONFIGS = {
    "single": [(0, 0)],
    "line2":  [(0, 0), (1, 0)],
    "line3":  [(0, 0), (1, 0), (2, 0)],
    "square4": [(0, 0), (1, 0), (0, 1), (1, 1)],
    # 以下為留出構型
    "L3":     [(0, 0), (1, 0), (0, 1)],
    "T4":     [(0, 0), (1, 0), (2, 0), (1, 1)],
    "line4":  [(0, 0), (1, 0), (2, 0), (3, 0)],
    "grid6":  [(i, j) for i in range(3) for j in range(2)],
}
P0_CONFIGS = dict(CONFIGS)   # P0 可控性／可辨識性分析只用這 8 個，輸出不因新增構型而改變
# P0b（感知半徑）新增的大構型
CONFIGS.update({
    "line6": [(i, 0) for i in range(6)],
    "line8": [(i, 0) for i in range(8)],
    "grid9": [(i, j) for i in range(3) for j in range(3)],
    "grid10": [(i, j) for i in range(5) for j in range(2)],   # x 方向 5、y 方向 2
    "plus5": [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)],        # 十字，中心 (1,1)
})
CONFIGS.update({     # G1 v2 新增的鑑別組構型（細長線）
    "line5": [(i, 0) for i in range(5)],
    "line7": [(i, 0) for i in range(7)],
})
LARGE = ["line6", "line8", "grid9", "grid10", "plus5"]
TRAIN = ["single", "line2", "line3", "square4"]

# 單一模組：馬達在 (±a, ±a)，轉向交錯（相鄰反向、對角同向）。
_a = CF2X["a"]
MOTOR_XY = np.array([[+_a, -_a], [-_a, -_a], [-_a, +_a], [+_a, +_a]])
MOTOR_SPIN = np.array([+1, -1, +1, -1])   # +1：反扭矩使機身 +z 轉（假設的符號慣例）


def assemble(cells, p=CF2X, pitch=MODULE_PITCH):
    """回傳 dict：N, M, com, J(3x3, 相對 CoM), rotor_pos(4N,3), rotor_spin(4N), A(4x4N)。"""
    c = np.asarray(cells, float) * pitch
    c = np.c_[c, np.zeros(len(c))]            # 模組中心 (N,3)，都在 z=0 平面
    N, m = len(c), p["mass"]
    com = c.mean(0)                            # 模組完全相同 → CoM 就是中心的平均
    d = c - com
    # 每個模組自己的慣量 + 平行軸定理 m(|d|²I − d dᵀ)
    J0 = np.diag([p["Ixx"], p["Iyy"], p["Izz"]])
    J = sum(J0 + m * (d_ @ d_ * np.eye(3) - np.outer(d_, d_)) for d_ in d)
    # 4N 顆轉子相對 CoM 的位置與轉向
    pos = np.vstack([d_ + np.r_[xy, 0] for d_ in d for xy in MOTOR_XY])
    spin = np.tile(MOTOR_SPIN, N)
    # A 把每顆推力映射成 [Fz, τx, τy, τz]；τx = y·T, τy = −x·T, τz = km·spin·T
    A = np.vstack([np.ones(4 * N), pos[:, 1], -pos[:, 0], p["km_ratio"] * spin])
    return dict(N=N, M=N * m, com=com, J=J, rotor_pos=pos, rotor_spin=spin, A=A)


def sym_matrix(asm):
    """symmetric 制度：每個模組輸出同樣 4 顆推力 u → 整架 T = tile(u, N)，A_sym = A @ tile (4x4)。"""
    return asm["A"] @ np.tile(np.eye(4), (asm["N"], 1))
