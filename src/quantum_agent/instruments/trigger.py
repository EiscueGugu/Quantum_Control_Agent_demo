"""虚拟触发源 (Trigger Source)。

包含:
    TriggerConfig   — 硬件配置参数 (从 instruments.yaml 加载)
    VirtualTrigger  — 虚拟 Trigger 实例 (加载配置, 模拟触发脉冲序列)
"""

import json
from dataclasses import dataclass

import yaml


@dataclass
class TriggerConfig:
    """Trigger 硬件配置参数。"""
    name: str                        # 仪器名称
    channels: int                    # 触发通道数
    pulse_width_ns: float            # 触发脉冲宽度 (ns)
    pulse_amplitude_V: float         # 触发脉冲幅度 (V, TTL 通常 5.0V)


def load_trigger_config(config_path: str) -> TriggerConfig:
    """从 instruments.yaml 加载 Trigger 配置段。"""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    t = data["trigger"]
    return TriggerConfig(
        name=t["name"],
        channels=int(t["channels"]),
        pulse_width_ns=float(t["pulse_width_ns"]),
        pulse_amplitude_V=float(t["pulse_amplitude_V"]),
    )


class VirtualTrigger:
    """虚拟触发源。

    模拟真实 Trigger 的行为:
        1. load_config() — 加载触发配置 JSON
        2. fire()        — 模拟触发序列，返回事件数

    触发信号通过物理线缆同时送给 VSG 和 DAQ 的触发输入，
    同步两者的波形播放和数据采集。
    """

    def __init__(self, config: TriggerConfig):
        self.config = config
        self._trig_config: dict = {}

    def load_config(self, config_path: str) -> None:
        """加载 Trigger 配置 JSON。"""
        with open(config_path, "r", encoding="utf-8") as f:
            self._trig_config = json.load(f)

    def fire(self) -> int:
        """模拟触发序列，返回触发事件总数。

        每个扫描点对应一个触发事件。
        """
        if not self._trig_config:
            raise RuntimeError("未加载触发配置，请先调用 load_config()")
        return self._trig_config.get("total_events", 0)

    @property
    def trig_config(self) -> dict:
        """返回已加载的触发配置。"""
        return dict(self._trig_config)
