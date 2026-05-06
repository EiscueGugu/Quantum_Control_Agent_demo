"""Tool4 — 预处理层：对原始数据进行预处理并存储。

当前为纯透传 (pass-through)，不做任何变换。
后续可在此扩展: 滤波、归一化、I/Q 分离、背景扣除等。
"""

from pathlib import Path
from typing import Optional

import numpy as np

from ..instruments.daq import VirtualDAQ
from ..models.processed_data import ProcessedData
from ..models.raw_data import RawData


class DataPreprocessor:
    """数据预处理器。当前实现为透传，直接复制原始数据到 processed/ 目录。"""

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)

    def process(self, raw: RawData, experiment_id: Optional[str] = None) -> ProcessedData:
        """透传原始数据并保存为 HDF5。

        Args:
            raw: Tool3 输出的原始数据
            experiment_id: 实验 ID，为 None 时自动生成

        Returns:
            ProcessedData 指向 processed/ 目录下的 HDF5 文件
        """
        eid = experiment_id or f"{raw.experiment_type}_{raw.qubit}"
        out_dir = self.output_dir / eid
        out_dir.mkdir(parents=True, exist_ok=True)

        data = raw.data.copy()  # 透传: 不做变换

        proc_path = str(out_dir / "processed_data.h5")
        VirtualDAQ.save(proc_path, {
            "data": data,
            "scan_values": np.array(raw.scan_values),
        })

        return ProcessedData(
            path=proc_path,
            experiment_type=raw.experiment_type,
            qubit=raw.qubit,
            scan_values=np.array(raw.scan_values),
            data=data,
            raw_data_path=raw.path,  # 溯源原始数据
            metadata={
                "source": "tool4_pass_through",
                "raw_data_path": raw.path,
                **raw.metadata,
            },
        )
