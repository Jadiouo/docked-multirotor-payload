"""讀 CAD 模組參數（cad/out/module_props.json），整理成 assemble() 與環境要用的三樣東西。
質量與慣量用 CAD 值；馬達（a、Tmax、km_ratio、tau）沿用 params.py 的 CF2X。
不修改 params.py：P0 的分析依賴它。"""
import json
from pathlib import Path
import numpy as np
from .params import CF2X

DEFAULT_PROPS = Path(__file__).resolve().parent.parent / "cad" / "out" / "module_props.json"


def load_module(path=DEFAULT_PROPS):
    """回傳 (p, pitch, joint_limits)。
    p：給 assemble(cells, p, pitch) 的 dict，鍵同 CF2X（mass、Ixx、Iyy、Izz、a、Tmax、km_ratio、tau）。
    pitch：相鄰模組中心距（m）。
    joint_limits：dict(pull_N、shear_N、peel_moment_Nm、…)。"""
    props = json.loads(Path(path).read_text())
    I = np.array(props["inertia"], float)
    # assemble() 假設模組 CoM 在框架中心、慣性張量是對角的；CAD 值若不符合，結果會悄悄出錯，所以直接擋下
    assert np.abs(props["com"]).max() < 1e-6, f"module_props 的 com 不在原點：{props['com']}"
    assert np.abs(I - np.diag(np.diag(I))).max() < 1e-3 * np.diag(I).max(), "module_props 的慣量不是對角的"
    p = dict(CF2X)                                        # a、Tmax、km_ratio、tau 用 CF2X
    p.update(mass=props["mass"], Ixx=float(I[0, 0]), Iyy=float(I[1, 1]), Izz=float(I[2, 2]))
    return p, props["pitch"], props["joint_limits"]
