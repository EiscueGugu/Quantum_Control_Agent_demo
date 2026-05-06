"""实验数据与拟合结果绘图。

plot_fit() 绘制散点数据 + 平滑拟合曲线，保存为 PNG。
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免弹窗
import matplotlib.pyplot as plt
import numpy as np


def plot_fit(
    x: np.ndarray,
    y: np.ndarray,
    y_fit: np.ndarray,
    experiment_type: str,
    save_path: str,
    model_fn=None,
    popt=None,
) -> None:
    """绘制实验数据散点和拟合曲线。

    Args:
        x: 扫描参数值 (ns)
        y: 实验数据
        y_fit: 在原始 x 点处的拟合值 (用于 fallback)
        experiment_type: 实验类型标签
        save_path: PNG 输出路径
        model_fn: 拟合函数 (可选)。传入时用 500 点密度网格绘制平滑曲线。
        popt: 拟合参数 (可选)。与 model_fn 搭配使用。
    """
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="Data", s=20, color="steelblue")

    if model_fn is not None and popt is not None:
        # 用 500 点密度生成平滑拟合曲线
        x_smooth = np.linspace(x[0], x[-1], 500)
        y_smooth = model_fn(x_smooth, *popt)
        plt.plot(x_smooth, y_smooth, "r-", linewidth=2, label="Fit")
    else:
        plt.plot(x, y_fit, "r-", linewidth=2, label="Fit")

    plt.xlabel("Scan parameter (ns)")
    plt.ylabel("Signal")
    plt.title(f"{experiment_type.upper()} Fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(p), dpi=150)
    plt.close()
