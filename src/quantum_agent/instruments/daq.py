"""虚拟数据采集卡 (Data Acquisition)。

包含:
    DAQConfig   — 硬件配置参数 (从 instruments.yaml 加载)
    VirtualDAQ  — 虚拟 DAQ 实例 (加载配置, 模拟采集, 保存 HDF5)
    _simulate_data — 根据实验类型生成带噪声的物理曲线
"""

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import yaml


@dataclass
class DAQConfig:
    """DAQ 硬件配置参数。"""
    name: str                        # 仪器名称
    sample_rate_Hz: float            # 采样率 (Hz), 默认 1.0 GHz
    channels: int                    # 通道数
    resolution_bits: int             # ADC 分辨率 (bit)
    voltage_range_V: tuple           # 电压范围 (min, max) V


def load_daq_config(config_path: str) -> DAQConfig:
    """从 instruments.yaml 加载 DAQ 配置段。"""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    daq = data["daq"]
    return DAQConfig(
        name=daq["name"],
        sample_rate_Hz=float(daq["sample_rate_Hz"]),
        channels=int(daq["channels"]),
        resolution_bits=int(daq["resolution_bits"]),
        voltage_range_V=tuple(daq["voltage_range_V"]),
    )


class VirtualDAQ:
    """虚拟数据采集卡。

    模拟真实 DAQ 的行为:
        1. load_config() — 加载采集配置 JSON
        2. arm()         — 进入外部触发等待模式
        3. acquire()     — 根据实验类型生成模拟数据
        4. save()        — 保存为 HDF5 (.h5)

    Usage:
        daq = VirtualDAQ(config)
        daq.load_config("daq_config.json")
        daq.arm()
        data = daq.acquire("rabi", scan_values)
        daq.save("raw_data.h5", {"data": data, "scan_values": scan_values})
    """

    def __init__(self, config: DAQConfig):
        self.config = config
        self._acq_config: dict = {}
        self._armed = False

    def load_config(self, config_path: str) -> None:
        """加载 DAQ 采集配置 JSON。"""
        with open(config_path, "r", encoding="utf-8") as f:
            self._acq_config = json.load(f)

    def arm(self) -> None:
        """进入外部触发等待模式。必须先加载配置。"""
        if not self._acq_config:
            raise RuntimeError("未加载采集配置，请先调用 load_config()")
        self._armed = True

    def acquire(self, experiment_type: str, scan_values: np.ndarray) -> np.ndarray:
        """根据实验类型生成模拟采集数据 (带噪声物理曲线)。

        Args:
            experiment_type: "rabi" / "t1" / "ramsey"
            scan_values: 扫描参数值 (ns)
        """
        if not self._armed:
            raise RuntimeError("DAQ 未 arm，请先调用 arm()")
        return _simulate_data(experiment_type, scan_values)

    @staticmethod
    def save(path: str, data_dict: dict) -> None:
        """保存数据为 HDF5 格式。

        - np.ndarray → dataset (gzip 压缩)
        - list/tuple → 转 np.ndarray 后保存
        - int/float/str/bool → attribute
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(p, "w") as f:
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value, compression="gzip")
                elif isinstance(value, (list, tuple)):
                    arr = np.array(value)
                    f.create_dataset(key, data=arr, compression="gzip")
                elif isinstance(value, (int, float, str, bool)):
                    f.attrs[key] = value

    @property
    def acq_config(self) -> dict:
        """返回已加载的采集配置。"""
        return dict(self._acq_config)

    @property
    def is_armed(self) -> bool:
        """是否已进入等待触发状态。"""
        return self._armed


# ── 物理模拟 ────────────────────────────────────────────────

def _simulate_data(experiment_type: str, scan_values: np.ndarray) -> np.ndarray:
    """根据实验类型和扫描参数生成带噪声的虚拟数据。

    噪声: N(0, 0.03) 高斯白噪声，随机种子固定 (42) 保证可复现。

    模拟公式:
        Rabi:   rabi_model(t, A=1.0, T_rabi=1.0us, phi=0, y0=0)
        T1:     exp(-t / 20us)
        Ramsey: 0.5 * (1 + cos(2π*2MHz*t + 0.3)) * exp(-t / 15us)
    """
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.03, len(scan_values))
    t = scan_values / 1000.0  # ns → us (模拟公式以 us 为单位)

    if experiment_type == "rabi":
        from ..utils.fitting_models import rabi_model
        signal = rabi_model(t, 1.0, 1.0, 0.0, 0.0)  # A=1.0, T_rabi=1.0us

    elif experiment_type == "t1":
        t1_us = 20.0
        signal = np.exp(-t / t1_us)

    elif experiment_type == "ramsey":
        from ..utils.fitting_models import ramsey_model
        # 参数: A, T2_star(ns), f_detune(GHz), phi, y0
        signal = ramsey_model(scan_values, 0.5, 30000.0, 0.0005, 0.3, 0.5)

    else:
        raise ValueError(f"不支持的实验类型: '{experiment_type}'")

    result = signal + noise
    return np.clip(result, 0.0, 1.0)
