import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from quantum_agent.instruments.daq import DAQConfig, VirtualDAQ
from quantum_agent.instruments.trigger import TriggerConfig, VirtualTrigger
from quantum_agent.instruments.vsg import VSGConfig, VirtualVSG
from quantum_agent.models.compiled_experiment import CompiledExperiment
from quantum_agent.models.raw_data import RawData
from quantum_agent.tools.tool3_executor import ExperimentExecutor


# --- Helpers ---

def _make_config() -> str:
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


def _make_executor(tmp_path: str) -> ExperimentExecutor:
    config_path = Path(tmp_path) / "instruments.yaml"
    config_path.write_text(_make_config(), encoding="utf-8")
    return ExperimentExecutor(str(config_path))


def _make_compiled(tmp_path: str, experiment_type: str, scan_values: list) -> CompiledExperiment:
    work_dir = Path(tmp_path) / "compiled" / f"test_{experiment_type}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy VSG waveform
    wf_path = str(work_dir / "vsg_waveform.npy")
    np.save(wf_path, np.zeros(100, dtype=np.complex128))

    # Create dummy metadata
    meta_path = str(work_dir / "vsg_metadata.json")
    meta = {"experiment_type": experiment_type, "qubit": "a", "scan_name": "x"}
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    # Create dummy DAQ config
    daq_config = {
        "sample_rate_Hz": 1.0e9,
        "channels": 4,
        "voltage_range_V": [-1.0, 1.0],
        "total_samples": 1000,
        "trigger_edge": "rising",
        "trigger_delay_ns": 0,
        "acquisition_windows": [{"scan_point": 0, "start_sample": 0, "duration_samples": 100}],
    }
    daq_path = str(work_dir / "daq_config.json")
    with open(daq_path, "w") as f:
        json.dump(daq_config, f)

    # Create Trigger config
    trig_config = {
        "trigger_channel": 0,
        "pulse_width_ns": 100,
        "pulse_amplitude_V": 5.0,
        "trigger_events": [{"scan_point": i, "timing_sample": 0} for i in range(len(scan_values))],
        "total_events": len(scan_values),
    }
    trig_path = str(work_dir / "trigger_config.json")
    with open(trig_path, "w") as f:
        json.dump(trig_config, f)

    return CompiledExperiment(
        experiment_id=f"test_{experiment_type}",
        work_dir=str(work_dir),
        vsg_waveform_path=wf_path,
        vsg_metadata_path=meta_path,
        daq_config_path=daq_path,
        trigger_config_path=trig_path,
        scan_values=scan_values,
        total_samples=1000,
        metadata={"experiment_type": experiment_type, "qubit": "a"},
    )


# --- VirtualVSG tests ---

def test_vsg_load_and_arm():
    vsg = VirtualVSG(VSGConfig("t", 2.4e9, 2, (100, 12000), (-30, 10)))
    with pytest.raises(RuntimeError, match="未加载波形"):
        vsg.arm()


def test_vsg_waveform_without_load_raises():
    vsg = VirtualVSG(VSGConfig("t", 2.4e9, 2, (100, 12000), (-30, 10)))
    with pytest.raises(RuntimeError, match="波形未加载"):
        _ = vsg.waveform


# --- VirtualDAQ tests ---

def test_daq_arm_without_config_raises():
    daq = VirtualDAQ(DAQConfig("t", 1e9, 4, 14, (-1, 1)))
    with pytest.raises(RuntimeError, match="未加载"):
        daq.arm()


def test_daq_acquire_without_arm_raises(tmp_path):
    daq = VirtualDAQ(DAQConfig("t", 1e9, 4, 14, (-1, 1)))
    cfg_path = Path(tmp_path) / "daq.json"
    cfg_path.write_text(json.dumps({"acquisition_windows": [], "total_samples": 100}))
    daq.load_config(str(cfg_path))
    with pytest.raises(RuntimeError, match="未 arm"):
        daq.acquire("rabi", np.array([1.0, 2.0]))


# --- VirtualTrigger tests ---

def test_trigger_fire_without_config_raises():
    trigger = VirtualTrigger(TriggerConfig("t", 2, 100, 5.0))
    with pytest.raises(RuntimeError, match="未加载"):
        trigger.fire()


# --- Simulated data tests ---

def _arm_daq(daq: VirtualDAQ, tmpdir: str) -> None:
    cfg_path = Path(tmpdir) / "daq_cfg.json"
    cfg_path.write_text(json.dumps({"acquisition_windows": [], "total_samples": 100}))
    daq.load_config(str(cfg_path))
    daq.arm()


def test_simulate_rabi_output_shape(tmp_path):
    daq = VirtualDAQ(DAQConfig("t", 1e9, 4, 14, (-1, 1)))
    _arm_daq(daq, str(tmp_path))
    scan = np.linspace(0, 2000, 50)  # ns
    data = daq.acquire("rabi", scan)
    assert len(data) == 50
    assert data.dtype == np.float64
    assert 0 <= data.max() <= 1.1


def test_simulate_t1_decays(tmp_path):
    daq = VirtualDAQ(DAQConfig("t", 1e9, 4, 14, (-1, 1)))
    _arm_daq(daq, str(tmp_path))
    scan = np.linspace(0, 50000, 100)  # 0-50 us
    data = daq.acquire("t1", scan)
    # T1 decay: later values should generally be smaller than early
    assert data[0] > data[-1]


def test_simulate_ramsey_oscillates(tmp_path):
    daq = VirtualDAQ(DAQConfig("t", 1e9, 4, 14, (-1, 1)))
    _arm_daq(daq, str(tmp_path))
    scan = np.linspace(0, 10000, 200)
    data = daq.acquire("ramsey", scan)
    # Should have oscillations (not just monotonic)
    diffs = np.diff(data)
    assert np.any(diffs > 0) and np.any(diffs < 0)


def test_simulate_unknown_type_raises(tmp_path):
    daq = VirtualDAQ(DAQConfig("t", 1e9, 4, 14, (-1, 1)))
    _arm_daq(daq, str(tmp_path))
    with pytest.raises(ValueError, match="不支持"):
        daq.acquire("unknown", np.array([1.0]))


# --- ExperimentExecutor tests ---

def test_execute_rabi(tmp_path):
    executor = _make_executor(str(tmp_path))
    compiled = _make_compiled(str(tmp_path), "rabi", [0, 500, 1000, 1500, 2000])
    raw = executor.execute(compiled)

    assert isinstance(raw, RawData)
    assert raw.experiment_type == "rabi"
    assert raw.qubit == "a"
    assert len(raw.data) == 5
    assert Path(raw.path).exists()
    assert raw.path.endswith(".h5")


def test_execute_t1(tmp_path):
    executor = _make_executor(str(tmp_path))
    compiled = _make_compiled(str(tmp_path), "t1", [0, 2500, 5000])
    raw = executor.execute(compiled)

    assert raw.experiment_type == "t1"
    assert len(raw.data) == 3
    assert raw.data[0] > raw.data[-1]


def test_execute_ramsey(tmp_path):
    executor = _make_executor(str(tmp_path))
    compiled = _make_compiled(str(tmp_path), "ramsey", np.linspace(0, 10000, 10).tolist())
    raw = executor.execute(compiled)

    assert raw.experiment_type == "ramsey"
    assert len(raw.data) == 10


def test_execute_trigger_mismatch_raises(tmp_path):
    executor = _make_executor(str(tmp_path))
    compiled = _make_compiled(str(tmp_path), "rabi", [0, 500, 1000])
    # Overwrite trigger config with wrong count
    trig_path = compiled.trigger_config_path
    trig = json.loads(Path(trig_path).read_text())
    trig["total_events"] = 99
    Path(trig_path).write_text(json.dumps(trig))

    with pytest.raises(RuntimeError, match="不匹配"):
        executor.execute(compiled)


# --- HDF5 tests ---

def test_h5_file_structure(tmp_path):
    executor = _make_executor(str(tmp_path))
    compiled = _make_compiled(str(tmp_path), "rabi", [0, 500, 1000])
    raw = executor.execute(compiled)

    with h5py.File(raw.path, "r") as f:
        assert "data" in f
        assert "scan_values" in f
        assert np.allclose(f["scan_values"][:], [0, 500, 1000])


def test_h5_preserves_data_types(tmp_path):
    executor = _make_executor(str(tmp_path))
    compiled = _make_compiled(str(tmp_path), "t1", [0.0, 2500.0, 5000.0])
    raw = executor.execute(compiled)

    with h5py.File(raw.path, "r") as f:
        assert f["data"].dtype == np.float64
        assert len(f["data"]) == 3
