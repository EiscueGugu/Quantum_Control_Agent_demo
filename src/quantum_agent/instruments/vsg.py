"""虚拟矢量信号发生器 (Vector Signal Generator)。

包含:
    VSGConfig   — 硬件配置参数 (从 instruments.yaml 加载)
    VirtualVSG  — 虚拟 VSG 实例 (加载波形, 等待触发)
"""

import json
from dataclasses import dataclass

import numpy as np
import yaml


@dataclass
class VSGConfig:
    """VSG 硬件配置参数。"""
    name: str                        # 仪器名称
    sample_rate_Hz: float            # 采样率 (Hz), 默认 2.4 GHz
    channels: int                    # 通道数
    frequency_range_MHz: tuple       # 频率范围 (min, max) MHz
    amplitude_range_dBm: tuple       # 幅度范围 (min, max) dBm


def load_vsg_config(config_path: str) -> VSGConfig:
    """从 instruments.yaml 加载 VSG 配置段。"""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    vsg = data["vsg"]
    return VSGConfig(
        name=vsg["name"],
        sample_rate_Hz=float(vsg["sample_rate_Hz"]),
        channels=int(vsg["channels"]),
        frequency_range_MHz=tuple(vsg["frequency_range_MHz"]),
        amplitude_range_dBm=tuple(vsg["amplitude_range_dBm"]),
    )


class VirtualVSG:
    """虚拟矢量信号发生器。

    模拟真实 VSG 的行为:
        1. load_experiment() — 加载 .npy 波形文件和 .json 元数据
        2. arm()              — 进入外部触发等待模式
        3. 触发后输出波形     — (虚拟实现，无实际信号)

    Usage:
        vsg = VirtualVSG(config)
        vsg.load_experiment("waveform.npy", "metadata.json")
        vsg.arm()
        wf = vsg.waveform
    """

    def __init__(self, config: VSGConfig):
        self.config = config
        self._waveform: np.ndarray | None = None
        self._metadata: dict = {}
        self._armed = False

    def load_experiment(self, waveform_path: str, metadata_path: str) -> None:
        """加载波形文件和元数据。

        Args:
            waveform_path: .npy 文件路径 (complex128 IQ 波形)
            metadata_path: .json 元数据 (实验参数、扫描信息)
        """
        self._waveform = np.load(waveform_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

    def arm(self) -> None:
        """进入外部触发等待模式。必须先加载波形。"""
        if self._waveform is None:
            raise RuntimeError("未加载波形文件，请先调用 load_experiment()")
        self._armed = True

    @property
    def waveform(self) -> np.ndarray:
        """返回已加载的 IQ 波形 (complex128)。"""
        if self._waveform is None:
            raise RuntimeError("波形未加载")
        return self._waveform

    @property
    def metadata(self) -> dict:
        """返回实验元数据副本。"""
        return dict(self._metadata)

    @property
    def is_armed(self) -> bool:
        """是否已进入等待触发状态。"""
        return self._armed
