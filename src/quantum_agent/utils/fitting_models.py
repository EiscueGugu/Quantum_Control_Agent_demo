"""量子测控实验拟合模型函数。

每个函数接受 (x, *params)，返回 y 数组。
x 单位为 ns，与扫描参数一致。
"""

import numpy as np


def rabi_model(x: np.ndarray, A: float, T_rabi: float, phi: float, y0: float) -> np.ndarray:
    """拉比振荡: A * sin(pi * x / T_rabi + phi)^2 + y0

    参数:
        A:      振荡幅度
        T_rabi: 拉比周期 (ns)
        phi:    初始相位 (rad)
        y0:    基线偏移
    """
    return A * np.sin(np.pi * x / T_rabi + phi) ** 2 + y0


def t1_model(x: np.ndarray, A: float, T1: float, y0: float) -> np.ndarray:
    """能量弛豫: A * exp(-x / T1) + y0

    参数:
        A:  初始幅度
        T1: 弛豫时间 (ns)
        y0: 基线偏移
    """
    return A * np.exp(-x / T1) + y0


def ramsey_model(
    x: np.ndarray, A: float, T2_star: float, f_detune: float, phi: float, y0: float
) -> np.ndarray:
    """Ramsey 干涉: A * exp(-x/T2_star) * cos(2*pi*f_detune*x + phi) + y0

    参数:
        A:        振荡幅度
        T2_star:  退相干时间 (ns)
        f_detune: 失谐频率 (GHz)
        phi:      初始相位 (rad)
        y0:       基线偏移
    """
    return A * np.exp(-x / T2_star) * np.cos(2.0 * np.pi * f_detune * x + phi) + y0
