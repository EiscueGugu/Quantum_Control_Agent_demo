"""Tool3 — 执行与采集层：驱动三个虚拟仪器 (VSG/DAQ/Trigger) 执行实验。

执行管线 (9 步):
    1. 创建虚拟仪器实例 (VSG, DAQ, Trigger)
    2. 加载所有配置文件
    3. Arm VSG (等待触发)
    4. Arm DAQ (等待触发)
    5. 启动 Trigger 序列
    6. 校验触发事件数
    7. DAQ 模拟采集 (根据实验类型生成物理公式数据)
    8. 保存为 HDF5 (.h5)
    9. 返回 RawData
"""

from pathlib import Path
from typing import Optional

import numpy as np

from ..instruments.daq import DAQConfig, VirtualDAQ, load_daq_config
from ..instruments.trigger import TriggerConfig, VirtualTrigger, load_trigger_config
from ..instruments.vsg import VSGConfig, VirtualVSG, load_vsg_config
from ..models.compiled_experiment import CompiledExperiment
from ..models.raw_data import RawData


class ExperimentExecutor:
    """驱动虚拟仪器执行实验并采集数据。"""

    def __init__(self, config_path: str = "config/instruments.yaml"):
        self.vsg_config = load_vsg_config(config_path)
        self.daq_config = load_daq_config(config_path)
        self.trigger_config = load_trigger_config(config_path)

    def execute(self, compiled: CompiledExperiment, output_dir: Optional[str] = None) -> RawData:
        """执行完整采集管线。

        Args:
            compiled: Tool2 输出的编译结果
            output_dir: 原始数据输出目录，默认使用 compiled.work_dir
        """
        out_dir = Path(output_dir or compiled.work_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. 创建虚拟仪器实例
        vsg = VirtualVSG(self.vsg_config)
        daq = VirtualDAQ(self.daq_config)
        trigger = VirtualTrigger(self.trigger_config)

        # 2. 加载所有配置
        vsg.load_experiment(compiled.vsg_waveform_path, compiled.vsg_metadata_path)
        daq.load_config(compiled.daq_config_path)
        trigger.load_config(compiled.trigger_config_path)

        # 3-4. Arm VSG 和 DAQ (进入外部触发等待模式)
        vsg.arm()
        daq.arm()

        # 5-6. 启动 Trigger，校验事件数
        n_events = trigger.fire()
        if n_events != len(compiled.scan_values):
            raise RuntimeError(
                f"触发事件数 ({n_events}) 与扫描点数 ({len(compiled.scan_values)}) 不匹配"
            )

        # 7. DAQ 模拟采集 (根据实验类型生成带噪声的物理曲线)
        experiment_type = compiled.metadata.get("experiment_type", "")
        scan_values = np.array(compiled.scan_values)
        data = daq.acquire(experiment_type, scan_values)

        # 8. 保存为 HDF5
        raw_path = str(out_dir / "raw_data.h5")
        daq.save(raw_path, {
            "data": data,
            "scan_values": scan_values,
        })

        return RawData(
            path=raw_path,
            experiment_type=experiment_type,
            qubit=compiled.metadata.get("qubit", ""),
            scan_values=scan_values,
            data=data,
            metadata={
                "sample_rate_Hz": self.daq_config.sample_rate_Hz,
                "trigger_events": n_events,
            },
        )
