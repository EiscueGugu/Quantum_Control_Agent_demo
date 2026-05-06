"""Tool2 输出 — 编译后的实验执行文件集合。"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PulseMarker:
    """单个脉冲在完整波形中的起止样本位置。"""
    pulse_name: str       # 脉冲名称
    start_sample: int     # 起始样本索引 (包含)
    end_sample: int       # 结束样本索引 (不包含)


@dataclass
class ScanPointResult:
    """一个扫描点的完整编译结果。"""
    index: int                        # 扫描点序号 (0-based)
    scan_value: float                 # 该点的扫描参数值 (单位: ns)
    pulse_markers: List[PulseMarker] = field(default_factory=list)  # 各脉冲起止位置


@dataclass
class CompiledExperiment:
    """Tool2 编译输出容器，包含 VSG/DAQ/Trigger 的完整执行文件路径。"""
    experiment_id: str                # 实验唯一 ID
    work_dir: str                     # 输出目录路径
    vsg_waveform_path: str            # VSG IQ 波形 (.npy, complex128)
    vsg_metadata_path: str            # VSG 元数据 (.json)
    daq_config_path: str              # DAQ 采集配置 (.json)
    trigger_config_path: str          # Trigger 触发配置 (.json)
    scan_values: List[float] = field(default_factory=list)          # 扫描参数值 (ns)
    scan_points: List[ScanPointResult] = field(default_factory=list)  # 每扫描点详情
    total_samples: int = 0            # 波形总采样点数
    metadata: Dict[str, Any] = field(default_factory=dict)          # 额外元数据
