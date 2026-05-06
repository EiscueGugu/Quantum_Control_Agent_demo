import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from quantum_agent.instruments.daq import DAQConfig, load_daq_config
from quantum_agent.instruments.trigger import TriggerConfig, load_trigger_config
from quantum_agent.instruments.vsg import VSGConfig, load_vsg_config
from quantum_agent.models.compiled_experiment import CompiledExperiment
from quantum_agent.models.experiment_ir import ExperimentIR, Pulse, ScanParameter
from quantum_agent.tools.tool2_compiler import ExperimentCompiler


# --- Helpers ---

def _make_config() -> str:
    """Create a temporary instruments.yaml for testing."""
    return """vsg:
  name: "Test VSG"
  sample_rate_Hz: 2.4e9
  channels: 2
  frequency_range_MHz: [100, 12000]
  amplitude_range_dBm: [-30, 10]

daq:
  name: "Test DAQ"
  sample_rate_Hz: 1.0e9
  channels: 4
  resolution_bits: 14
  voltage_range_V: [-1.0, 1.0]

trigger:
  name: "Test Trigger"
  channels: 2
  pulse_width_ns: 100
  pulse_amplitude_V: 5.0
"""


def _make_compiler(tmp_path: str) -> ExperimentCompiler:
    config_path = Path(tmp_path) / "instruments.yaml"
    config_path.write_text(_make_config(), encoding="utf-8")
    work_dir = str(Path(tmp_path) / "compiled")
    return ExperimentCompiler(str(config_path), work_dir)


def _make_rabi_ir() -> ExperimentIR:
    return ExperimentIR(
        experiment_type="rabi",
        qubit="a",
        fixed_params={"power_dBm": 10.0},
        scan=ScanParameter(name="pulse_duration", unit="us", start=0.0, stop=1.0, num_points=5),
        pulse=[
            Pulse(name="x180", shape="gaussian", duration_ns="$pulse_duration", amplitude=1.0),
            Pulse(name="readout", shape="square", duration_ns=1000, amplitude=1.0),
        ],
    )


def _make_t1_ir() -> ExperimentIR:
    return ExperimentIR(
        experiment_type="t1",
        qubit="b",
        fixed_params={},
        scan=ScanParameter(name="wait_time", unit="us", start=0.0, stop=5.0, num_points=3),
        pulse=[
            Pulse(name="x180", shape="gaussian", duration_ns=50, amplitude=1.0),
            Pulse(name="wait", shape="square", duration_ns="$wait_time", amplitude=0.0),
            Pulse(name="readout", shape="square", duration_ns=1000, amplitude=1.0),
        ],
    )


# --- Step 1: Expand scan ---

def test_expand_scan_us():
    compiler = ExperimentCompiler()
    result = compiler._expand_scan(0.0, 10.0, 11, "us")
    assert len(result) == 11
    assert result[0] == 0.0
    assert result[-1] == 10000.0  # 10 us = 10000 ns


def test_expand_scan_ns():
    compiler = ExperimentCompiler()
    result = compiler._expand_scan(0.0, 500.0, 6, "ns")
    assert len(result) == 6
    assert result[-1] == 500.0


def test_expand_scan_ms():
    compiler = ExperimentCompiler()
    result = compiler._expand_scan(0.0, 1.0, 3, "ms")
    assert result[0] == 0.0
    assert result[1] == 5e5  # 0.5 ms = 500000 ns
    assert result[2] == 1e6


# --- Step 2: Resolve placeholders ---

def test_resolve_placeholder_replaces_scan_var():
    compiler = ExperimentCompiler()
    pulses = [
        Pulse(name="x180", shape="gaussian", duration_ns="$pulse_duration", amplitude=1.0),
        Pulse(name="readout", shape="square", duration_ns=1000, amplitude=1.0),
    ]
    resolved = compiler._resolve_placeholders(pulses, "pulse_duration", 5000.0)
    assert resolved[0]["duration_ns"] == 5000.0  # placeholder replaced
    assert resolved[1]["duration_ns"] == 1000.0   # fixed value unchanged


def test_resolve_placeholder_mismatch_raises():
    compiler = ExperimentCompiler()
    pulses = [Pulse(name="x180", shape="gaussian", duration_ns="$wrong_name", amplitude=1.0)]
    with pytest.raises(ValueError, match="不匹配"):
        compiler._resolve_placeholders(pulses, "pulse_duration", 100.0)


# --- Step 3: Generate pulse waveform ---

def test_generate_gaussian_pulse():
    compiler = ExperimentCompiler()
    wf = compiler._generate_pulse_waveform("gaussian", 100.0, 1.0)  # 100 ns
    assert len(wf) > 0
    assert wf.dtype == np.float64
    # Max should be near amplitude=1.0 at center
    assert wf.max() == pytest.approx(1.0, abs=0.1)
    # Edges should be near 0
    assert wf[0] < 0.15


def test_generate_square_pulse():
    compiler = ExperimentCompiler()
    wf = compiler._generate_pulse_waveform("square", 100.0, 0.5)
    assert np.allclose(wf, 0.5)


# --- Step 4: Assemble waveform ---

def test_assemble_waveform_markers():
    compiler = ExperimentCompiler()
    pulses = [
        {"name": "x180", "shape": "gaussian", "duration_ns": 50, "amplitude": 1.0},
        {"name": "readout", "shape": "square", "duration_ns": 1000, "amplitude": 1.0},
    ]
    wf, markers = compiler._assemble_waveform(pulses)
    assert wf.dtype == np.complex128
    assert wf.ndim == 1
    assert len(markers) == 2
    assert markers[0]["name"] == "x180"
    assert markers[0]["start"] == 0
    assert markers[1]["start"] == markers[0]["end"]
    assert markers[1]["end"] == len(wf)


def test_wait_pulse_is_zeros():
    compiler = ExperimentCompiler()
    pulses = [{"name": "wait", "shape": "square", "duration_ns": 500, "amplitude": 0.0}]
    wf, _ = compiler._assemble_waveform(pulses)
    assert np.all(np.abs(wf) == 0.0)


# --- Full compile tests ---

def test_compile_rabi(tmp_path):
    compiler = _make_compiler(str(tmp_path))
    ir = _make_rabi_ir()
    result = compiler.compile(ir, "test_rabi")

    assert isinstance(result, CompiledExperiment)
    assert result.experiment_id == "test_rabi"
    assert len(result.scan_values) == 5
    assert result.total_samples > 0
    assert len(result.scan_points) == 5

    assert Path(result.vsg_waveform_path).exists()
    assert Path(result.vsg_waveform_path).suffix == ".npy"
    assert Path(result.vsg_metadata_path).exists()
    assert Path(result.daq_config_path).exists()
    assert Path(result.trigger_config_path).exists()

    # Verify waveform loads
    wf = np.load(result.vsg_waveform_path)
    assert wf.dtype == np.complex128
    assert len(wf) == result.total_samples


def test_compile_t1(tmp_path):
    compiler = _make_compiler(str(tmp_path))
    ir = _make_t1_ir()
    result = compiler.compile(ir, "test_t1")

    assert result.experiment_id == "test_t1"
    assert len(result.scan_values) == 3
    assert len(result.scan_points) == 3
    assert Path(result.vsg_waveform_path).exists()
    assert Path(result.trigger_config_path).exists()

    # T1 has 3 pulses: x180, wait, readout
    for sp in result.scan_points:
        names = [m.pulse_name for m in sp.pulse_markers]
        assert names == ["x180", "wait", "readout"]


def test_compile_auto_generates_id(tmp_path):
    compiler = _make_compiler(str(tmp_path))
    ir = _make_rabi_ir()
    result = compiler.compile(ir)
    assert result.experiment_id.startswith("rabi_a_")
    assert len(result.experiment_id) > 8


# --- VSG metadata ---

def test_vsg_metadata_contains_scan_info(tmp_path):
    compiler = _make_compiler(str(tmp_path))
    ir = _make_rabi_ir()
    result = compiler.compile(ir, "test_meta")

    with open(result.vsg_metadata_path, "r") as f:
        meta = json.load(f)

    assert meta["experiment_type"] == "rabi"
    assert meta["qubit"] == "a"
    assert meta["scan_name"] == "pulse_duration"
    assert len(meta["scan_values"]) == 5
    assert meta["total_samples"] == result.total_samples


# --- DAQ config ---

def test_daq_config_has_readout_windows(tmp_path):
    compiler = _make_compiler(str(tmp_path))
    ir = _make_rabi_ir()
    result = compiler.compile(ir, "test_daq")

    with open(result.daq_config_path, "r") as f:
        daq = json.load(f)

    assert daq["sample_rate_Hz"] == 1.0e9
    assert len(daq["acquisition_windows"]) == 5  # one per scan point (readout)
    for w in daq["acquisition_windows"]:
        assert w["duration_samples"] > 0


# --- Trigger config ---

def test_trigger_config_has_events(tmp_path):
    compiler = _make_compiler(str(tmp_path))
    ir = _make_rabi_ir()
    result = compiler.compile(ir, "test_trig")

    with open(result.trigger_config_path, "r") as f:
        trig = json.load(f)

    assert trig["pulse_width_ns"] == 100
    assert trig["pulse_amplitude_V"] == 5.0
    assert trig["total_events"] == 5
    assert len(trig["trigger_events"]) == 5
    assert trig["trigger_events"][0]["timing_sample"] == 0


def test_trigger_config_events_spacing(tmp_path):
    compiler = _make_compiler(str(tmp_path))
    ir = _make_rabi_ir()
    result = compiler.compile(ir, "test_spacing")

    with open(result.trigger_config_path, "r") as f:
        trig = json.load(f)

    for i, ev in enumerate(trig["trigger_events"]):
        assert ev["scan_point"] == i


# --- Config loading ---

def test_load_vsg_config(tmp_path):
    config_path = Path(tmp_path) / "instruments.yaml"
    config_path.write_text(_make_config(), encoding="utf-8")
    cfg = load_vsg_config(str(config_path))
    assert isinstance(cfg, VSGConfig)
    assert cfg.sample_rate_Hz == 2.4e9
    assert cfg.channels == 2


def test_load_daq_config(tmp_path):
    config_path = Path(tmp_path) / "instruments.yaml"
    config_path.write_text(_make_config(), encoding="utf-8")
    cfg = load_daq_config(str(config_path))
    assert isinstance(cfg, DAQConfig)
    assert cfg.sample_rate_Hz == 1.0e9
    assert cfg.resolution_bits == 14


def test_load_trigger_config(tmp_path):
    config_path = Path(tmp_path) / "instruments.yaml"
    config_path.write_text(_make_config(), encoding="utf-8")
    cfg = load_trigger_config(str(config_path))
    assert isinstance(cfg, TriggerConfig)
    assert cfg.channels == 2
    assert cfg.pulse_width_ns == 100
    assert cfg.pulse_amplitude_V == 5.0
