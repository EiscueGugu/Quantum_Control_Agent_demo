"""Tool1～Tool5 之间传递的标准化数据结构 — 实验中间表示 (Experiment IR)。"""

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class ScanParameter:
    """扫描参数定义，描述实验中需要变化的物理量。

    scan.name 与 pulse 中的 $占位符 对应。
    例如 scan.name="pulse_duration" 对应 pulse 中 duration_ns="$pulse_duration"。
    """
    name: str           # 参数名，如 "pulse_duration", "wait_time", "free_evolution_time"
    unit: str           # 单位: "us", "ns", "ms", "s"
    start: float        # 扫描起始值
    stop: float         # 扫描终止值
    num_points: int     # 扫描点数
    linear: bool = True # True=线性扫描, False=对数扫描 (预留)


@dataclass
class Pulse:
    """单个脉冲的定义。

    duration_ns 有两种形式:
        - float: 固定时长 (ns)，如 50.0 表示 50 ns 的 π 脉冲
        - str:   扫描占位符，如 "$pulse_duration" 表示该脉冲时长随扫描变化
    """
    name: str           # 标准脉冲名: x180, x90, x90y, readout, wait
    shape: str          # 波形形状: "gaussian" (高斯), "square" (方波)
    duration_ns: float | str  # ns 数值或 $占位符
    amplitude: float    # 归一化幅度 [0, 1]; wait 脉冲通常为 0.0


@dataclass
class ExperimentIR:
    """实验中间表示，Tool1 输出、Tool2 输入的核心数据结构。

    包含完整的实验描述: 类型、目标量子比特、固定参数、扫描参数、脉冲序列。
    """
    experiment_type: str                # "rabi" | "t1" | "ramsey"
    qubit: str                          # 量子比特标识, 如 "a", "b", "q1"
    scan: ScanParameter                 # 扫描参数
    fixed_params: Dict[str, Any] = field(default_factory=dict)  # 固定参数 {"power_dBm": 10.0, ...}
    pulse: List[Pulse] = field(default_factory=list)            # 脉冲序列 (按时序排列)
