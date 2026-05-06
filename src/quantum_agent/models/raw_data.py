"""Tool3 输出 — 虚拟仪器采集的原始数据。"""

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class RawData:
    """原始实验数据容器，由虚拟仪器采集生成。

    data 与 scan_values 一一对应: data[i] 是 scan_values[i] 扫描点处的采集信号。
    """
    path: str                      # .h5 文件路径
    experiment_type: str           # "rabi" / "t1" / "ramsey"
    qubit: str                     # 量子比特标识
    scan_values: np.ndarray        # 扫描参数值 (1D, 单位 ns)
    data: np.ndarray               # 采集信号 (1D, 长度同 scan_values)
    metadata: Dict[str, Any] = field(default_factory=dict)  # sample_rate_Hz, trigger_events 等
