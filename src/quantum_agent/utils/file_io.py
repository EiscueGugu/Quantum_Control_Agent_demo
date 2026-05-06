"""ExperimentIR 的 JSON 序列化/反序列化工具。

用于在 Tool 之间持久化实验中间表示。
"""

import json
from pathlib import Path
from typing import Any, Dict

from ..models.experiment_ir import ExperimentIR, ScanParameter, Pulse


def _ir_to_dict(ir: ExperimentIR) -> Dict[str, Any]:
    """ExperimentIR → JSON 可序列化的 dict。"""
    return {
        "experiment_type": ir.experiment_type,
        "qubit": ir.qubit,
        "fixed_params": ir.fixed_params,
        "scan": {
            "name": ir.scan.name,
            "unit": ir.scan.unit,
            "start": ir.scan.start,
            "stop": ir.scan.stop,
            "num_points": ir.scan.num_points,
            "linear": ir.scan.linear,
        },
        "pulse": [
            {
                "name": p.name,
                "shape": p.shape,
                "duration_ns": p.duration_ns,
                "amplitude": p.amplitude,
            }
            for p in ir.pulse
        ],
    }


def _dict_to_ir(data: Dict[str, Any]) -> ExperimentIR:
    """dict → ExperimentIR 反序列化。"""
    scan_data = data["scan"]
    scan = ScanParameter(
        name=scan_data["name"],
        unit=scan_data["unit"],
        start=float(scan_data["start"]),
        stop=float(scan_data["stop"]),
        num_points=int(scan_data["num_points"]),
        linear=scan_data.get("linear", True),
    )
    pulses = [
        Pulse(
            name=p["name"],
            shape=p["shape"],
            duration_ns=p["duration_ns"],
            amplitude=float(p["amplitude"]),
        )
        for p in data["pulse"]
    ]
    return ExperimentIR(
        experiment_type=data["experiment_type"],
        qubit=data["qubit"],
        fixed_params=data.get("fixed_params", {}),
        scan=scan,
        pulse=pulses,
    )


def save_experiment_ir(ir: ExperimentIR, path: str) -> None:
    """将 ExperimentIR 序列化为 JSON 文件。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(_ir_to_dict(ir), f, indent=2, ensure_ascii=False)


def load_experiment_ir(path: str) -> ExperimentIR:
    """从 JSON 文件反序列化为 ExperimentIR。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _dict_to_ir(data)
