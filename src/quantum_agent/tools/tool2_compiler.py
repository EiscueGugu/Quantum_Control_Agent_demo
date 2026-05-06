"""Tool2 — 编译层：将 ExperimentIR 编译为虚拟仪器可执行的底层文件和波形。

编译管线 (8 步):
    1. _expand_scan()          — 展开扫描参数，单位统一为 ns
    2. _resolve_placeholders() — 将 $占位符 替换为实际数值
    3. _generate_pulse_waveform() — 生成单个脉冲的 I 路包络波形
    4. _assemble_waveform()    — 拼接一个扫描点的所有脉冲
    5. 保存 VSG 文件            — .npy 波形 + .json 元数据
    6. 保存 DAQ 文件            — .json 采集配置
    7. 保存 Trigger 文件        — .json 触发配置
    8. 返回 CompiledExperiment — 包含所有输出路径和元数据
"""

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..instruments.daq import DAQConfig, load_daq_config
from ..instruments.trigger import TriggerConfig, load_trigger_config
from ..instruments.vsg import VSGConfig, load_vsg_config
from ..models.compiled_experiment import (
    CompiledExperiment,
    PulseMarker,
    ScanPointResult,
)
from ..models.experiment_ir import ExperimentIR, Pulse

# 单位 → ns 转换因子
_UNIT_TO_NS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}


class ExperimentCompiler:
    """将实验 IR 编译为仪器执行文件的编译器。

    初始化时加载 VSG/DAQ/Trigger 的硬件配置参数。
    """

    def __init__(self, config_path: str = "config/instruments.yaml", work_dir: str = "data/compiled"):
        self.vsg_config = load_vsg_config(config_path)
        self.daq_config = load_daq_config(config_path)
        self.trigger_config = load_trigger_config(config_path)
        self.work_dir = Path(work_dir)

    def compile(self, ir: ExperimentIR, experiment_id: Optional[str] = None) -> CompiledExperiment:
        """执行完整编译管线。

        Args:
            ir: Tool1 输出的实验中间表示
            experiment_id: 实验唯一 ID，为 None 时自动生成

        Returns:
            CompiledExperiment 包含所有输出文件路径
        """
        exp_id = experiment_id or f"{ir.experiment_type}_{ir.qubit}_{uuid.uuid4().hex[:8]}"
        exp_dir = self.work_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        scan_values_ns = self._expand_scan(ir.scan.start, ir.scan.stop, ir.scan.num_points, ir.scan.unit)

        scan_points: List[ScanPointResult] = []
        all_waveforms: List[np.ndarray] = []

        # 对每个扫描点生成完整波形
        for i, scan_val in enumerate(scan_values_ns):
            resolved = self._resolve_placeholders(ir.pulse, ir.scan.name, scan_val)
            wf, markers = self._assemble_waveform(resolved)
            all_waveforms.append(wf)
            scan_points.append(ScanPointResult(
                index=i,
                scan_value=scan_val,
                pulse_markers=[
                    PulseMarker(pulse_name=m["name"], start_sample=m["start"], end_sample=m["end"])
                    for m in markers
                ],
            ))

        full_waveform = np.concatenate(all_waveforms)
        total_samples = int(full_waveform.shape[0])

        # Step 5: VSG 波形 + 元数据
        vsg_wf_path = str(exp_dir / "vsg_waveform.npy")
        np.save(vsg_wf_path, full_waveform)

        vsg_meta = self._build_vsg_metadata(ir, scan_values_ns.tolist(), scan_points, total_samples)
        vsg_meta_path = str(exp_dir / "vsg_metadata.json")
        with open(vsg_meta_path, "w", encoding="utf-8") as f:
            json.dump(vsg_meta, f, indent=2, ensure_ascii=False)

        # Step 6: DAQ 采集配置
        daq_cfg = self._build_daq_config(scan_points, total_samples)
        daq_path = str(exp_dir / "daq_config.json")
        with open(daq_path, "w", encoding="utf-8") as f:
            json.dump(daq_cfg, f, indent=2, ensure_ascii=False)

        # Step 7: Trigger 触发配置
        trigger_cfg = self._build_trigger_config(scan_points)
        trigger_path = str(exp_dir / "trigger_config.json")
        with open(trigger_path, "w", encoding="utf-8") as f:
            json.dump(trigger_cfg, f, indent=2, ensure_ascii=False)

        return CompiledExperiment(
            experiment_id=exp_id,
            work_dir=str(exp_dir),
            vsg_waveform_path=vsg_wf_path,
            vsg_metadata_path=vsg_meta_path,
            daq_config_path=daq_path,
            trigger_config_path=trigger_path,
            scan_values=scan_values_ns.tolist(),
            scan_points=scan_points,
            total_samples=total_samples,
            metadata={"experiment_type": ir.experiment_type, "qubit": ir.qubit},
        )

    # ── Step 1: 展开扫描参数 ──────────────────────────────────

    def _expand_scan(self, start: float, stop: float, num_points: int, unit: str) -> np.ndarray:
        """将扫描范围展开为等间距数组，统一转换为 ns 单位。"""
        factor = _UNIT_TO_NS.get(unit, 1.0)
        return np.linspace(start * factor, stop * factor, num_points)

    # ── Step 2: 替换占位符 ────────────────────────────────────

    def _resolve_placeholders(self, pulses: List[Pulse], scan_name: str, scan_value_ns: float) -> List[Dict[str, Any]]:
        """将脉冲中的 $占位符 替换为当前扫描点的实际数值。

        "$pulse_duration" → 当前扫描值 (ns)
        固定数值保持不变。
        """
        resolved = []
        for p in pulses:
            entry = {"name": p.name, "shape": p.shape, "amplitude": float(p.amplitude)}

            dur = p.duration_ns
            if isinstance(dur, str) and dur.startswith("$"):
                param_name = dur[1:]  # 去掉 $
                if param_name == scan_name:
                    entry["duration_ns"] = scan_value_ns
                else:
                    raise ValueError(f"占位符 '{dur}' 与扫描参数名 '{scan_name}' 不匹配")
            else:
                entry["duration_ns"] = float(dur)
            resolved.append(entry)
        return resolved

    # ── Step 3: 生成单个脉冲波形 ─────────────────────────────

    def _generate_pulse_waveform(self, shape: str, duration_ns: float, amplitude: float) -> np.ndarray:
        """根据形状和参数生成单个脉冲的基带包络 (I 路, float64)。

        Gaussian: envelope = amplitude * exp(-0.5 * ((t-center)/sigma)^2), sigma = duration/4
        Square:   envelope = amplitude * ones(n_samples)
        """
        sample_rate_hz = self.vsg_config.sample_rate_Hz
        duration_s = duration_ns * 1e-9
        n_samples = max(1, int(round(duration_s * sample_rate_hz)))

        t = np.linspace(0, duration_ns, n_samples)

        if shape == "gaussian":
            center = duration_ns / 2.0
            sigma = duration_ns / 4.0  # 脉冲宽度 = 4σ，包络从 ~0 到峰值再回 ~0
            envelope = amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)
        elif shape == "square":
            envelope = amplitude * np.ones(n_samples)
        else:
            raise ValueError(f"不支持的脉冲形状: '{shape}'")

        return envelope.astype(np.float64)

    # ── Step 4: 拼接一个扫描点的完整波形 ────────────────────

    def _assemble_waveform(self, pulses: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """将一个扫描点的所有脉冲按顺序拼接为完整 IQ 波形。

        Returns:
            (full_waveform, markers): IQ 复数数组 + 每个脉冲的起止样本索引
        """
        segments = []
        markers: List[Dict[str, Any]] = []
        cursor = 0  # 当前样本位置

        for p in pulses:
            if p["amplitude"] == 0.0 or p["duration_ns"] == 0.0:
                # wait 脉冲或零时长 → 全零
                sample_rate_hz = self.vsg_config.sample_rate_Hz
                duration_s = float(p["duration_ns"]) * 1e-9
                n_samples = max(1, int(round(duration_s * sample_rate_hz)))
                wf = np.zeros(n_samples, dtype=np.float64)
            else:
                wf = self._generate_pulse_waveform(p["shape"], float(p["duration_ns"]), float(p["amplitude"]))

            n = int(wf.shape[0])
            segments.append(wf)
            markers.append({"name": p["name"], "start": cursor, "end": cursor + n})
            cursor += n

        full_wf = np.concatenate(segments) if segments else np.array([], dtype=np.float64)
        iq = full_wf + 0j  # 基带: I + jQ (Q 恒为零)
        return iq, markers

    # ── Step 5: VSG 元数据 ────────────────────────────────────

    def _build_vsg_metadata(
        self, ir: ExperimentIR, scan_values: List[float],
        scan_points: List[ScanPointResult], total_samples: int,
    ) -> Dict[str, Any]:
        """构建 VSG 元数据 JSON，记录实验参数和每个扫描点的脉冲起止位置。"""
        return {
            "experiment_type": ir.experiment_type,
            "qubit": ir.qubit,
            "sample_rate_Hz": self.vsg_config.sample_rate_Hz,
            "fixed_params": ir.fixed_params,
            "scan_name": ir.scan.name,
            "scan_unit": ir.scan.unit,
            "scan_values": scan_values,
            "total_samples": total_samples,
            "scan_points": [
                {
                    "index": sp.index,
                    "scan_value": sp.scan_value,
                    "pulse_markers": [
                        {"name": m.pulse_name, "start_sample": m.start_sample, "end_sample": m.end_sample}
                        for m in sp.pulse_markers
                    ],
                }
                for sp in scan_points
            ],
        }

    # ── Step 6: DAQ 采集配置 ──────────────────────────────────

    def _build_daq_config(self, scan_points: List[ScanPointResult], total_samples: int) -> Dict[str, Any]:
        """构建 DAQ 采集配置 JSON。

        自动识别所有 readout 脉冲的位置，生成对应的采集窗口列表。
        """
        acq_windows = []
        for sp in scan_points:
            for m in sp.pulse_markers:
                if m.pulse_name == "readout":
                    acq_windows.append({
                        "scan_point": sp.index,
                        "start_sample": m.start_sample,
                        "duration_samples": m.end_sample - m.start_sample,
                    })

        return {
            "sample_rate_Hz": self.daq_config.sample_rate_Hz,
            "channels": self.daq_config.channels,
            "voltage_range_V": list(self.daq_config.voltage_range_V),
            "total_samples": total_samples,
            "trigger_edge": "rising",
            "trigger_delay_ns": 0,
            "acquisition_windows": acq_windows,
        }

    # ── Step 7: Trigger 触发配置 ─────────────────────────────

    def _build_trigger_config(self, scan_points: List[ScanPointResult]) -> Dict[str, Any]:
        """为每个扫描点生成一个触发事件，触发时刻对齐到该扫描点的起始样本位置。"""
        trigger_events = []
        cursor = 0
        for sp in scan_points:
            trigger_events.append({
                "scan_point": sp.index,
                "timing_sample": cursor,
            })
            if sp.pulse_markers:
                sp_samples = sp.pulse_markers[-1].end_sample - sp.pulse_markers[0].start_sample
                cursor += sp_samples

        return {
            "trigger_channel": 0,
            "pulse_width_ns": self.trigger_config.pulse_width_ns,
            "pulse_amplitude_V": self.trigger_config.pulse_amplitude_V,
            "trigger_events": trigger_events,
            "total_events": len(trigger_events),
        }
