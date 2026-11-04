"""四旋翼參數。複製自 drone-flip-proprio/m1/params.py，2026-10-24（只取 cf2x，不含 perturb）。單位 SI。"""
G = 9.81

CF2X = dict(
    mass=0.027,                      # kg  gym-pybullet-drones cf2x.urdf；Menagerie 也是 0.027
    Ixx=1.4e-5, Iyy=1.4e-5, Izz=2.17e-5,   # gym-pybullet-drones cf2x.urdf (Menagerie 的值約 1.7 倍，見原報告)
    a=0.028,                         # m  馬達 x、y 偏移 cf2x.urdf prop link 位置
    Tmax=0.149,                      # N/馬達  由 thrust2weight=2.25: 2.25*0.027*9.81/4
    km_ratio=7.94e-12/3.16e-10,      # m   反扭矩/推力 = km/kf (cf2x.urdf)
    tau=0.07,                        # s   馬達一階時間常數，Eschmann 2024 fitted 0.072 / 廠商階躍 0.073
)

# 假設（可調）：相鄰模組中心距 0.10 m。cf2x 螺旋槳半徑約 0.023 m、馬達偏移 0.028 m，
# 模組外緣約 0.05 m，所以 0.10 m 剛好是「有保護框、槳不互相干涉」的最小值。
MODULE_PITCH = 0.10
