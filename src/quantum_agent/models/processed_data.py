"""Tool4 输出 — 预处理后的数据。"""

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class ProcessedData:
    """预处理数据容器。

    当前为纯透传 (与 RawData 相同)，后续可扩展滤波/归一化后的数据。
    raw_data_path 保留原始数据引用，用于溯源。
    """
    path: str                      # .h5 文件路径
    experiment_type: str           # "rabi" / "t1" / "ramsey"
    qubit: str                     # 量子比特标识
    scan_values: np.ndarray        # 扫描参数值 (1D, 单位 ns)
    data: np.ndarray               # 预处理后的信号 (1D)
    raw_data_path: str             # 引用原始数据的路径
    metadata: Dict[str, Any] = field(default_factory=dict)
