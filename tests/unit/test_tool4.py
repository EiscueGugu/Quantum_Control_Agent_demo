import json
from pathlib import Path

import h5py
import numpy as np

from quantum_agent.instruments.daq import VirtualDAQ, DAQConfig
from quantum_agent.models.processed_data import ProcessedData
from quantum_agent.models.raw_data import RawData
from quantum_agent.tools.tool4_preprocessor import DataPreprocessor


# --- Helpers ---

def _make_raw_data(tmp_path: str, experiment_type: str = "rabi",
                   qubit: str = "a", n_points: int = 5) -> RawData:
    out_dir = Path(tmp_path) / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_values = np.linspace(0, 2000, n_points)

    daq = VirtualDAQ(DAQConfig("t", 1e9, 4, 14, (-1, 1)))
    # Arm daq with a minimal config
    cfg_path = Path(tmp_path) / "daq.json"
    cfg_path.write_text(json.dumps({"acquisition_windows": [], "total_samples": 100}))
    daq.load_config(str(cfg_path))
    daq.arm()
    data = daq.acquire(experiment_type, scan_values)

    raw_path = str(out_dir / "raw_data.h5")
    daq.save(raw_path, {"data": data, "scan_values": scan_values})

    return RawData(
        path=raw_path,
        experiment_type=experiment_type,
        qubit=qubit,
        scan_values=scan_values,
        data=data,
        metadata={"sample_rate_Hz": 1e9, "trigger_events": n_points},
    )


# --- Process tests ---

def test_process_returns_processed_data(tmp_path):
    preprocessor = DataPreprocessor(str(Path(tmp_path) / "processed"))
    raw = _make_raw_data(str(tmp_path), "rabi", "a", 5)
    result = preprocessor.process(raw, "test_rabi")

    assert isinstance(result, ProcessedData)
    assert result.experiment_type == "rabi"
    assert result.qubit == "a"
    assert len(result.data) == 5
    assert len(result.scan_values) == 5


def test_process_preserves_data(tmp_path):
    preprocessor = DataPreprocessor(str(Path(tmp_path) / "processed"))
    raw = _make_raw_data(str(tmp_path), "t1", "b", 3)
    result = preprocessor.process(raw, "test_t1")

    assert np.allclose(result.data, raw.data)
    assert np.allclose(result.scan_values, raw.scan_values)


def test_process_writes_h5_file(tmp_path):
    preprocessor = DataPreprocessor(str(Path(tmp_path) / "processed"))
    raw = _make_raw_data(str(tmp_path), "ramsey", "a", 4)
    result = preprocessor.process(raw, "test_ramsey")

    assert Path(result.path).exists()
    assert result.path.endswith(".h5")


def test_process_h5_structure(tmp_path):
    preprocessor = DataPreprocessor(str(Path(tmp_path) / "processed"))
    raw = _make_raw_data(str(tmp_path), "rabi", "a", 5)
    result = preprocessor.process(raw, "test_struct")

    with h5py.File(result.path, "r") as f:
        assert "data" in f
        assert "scan_values" in f
        assert len(f["data"]) == 5
        assert len(f["scan_values"]) == 5


def test_process_metadata_references_raw(tmp_path):
    preprocessor = DataPreprocessor(str(Path(tmp_path) / "processed"))
    raw = _make_raw_data(str(tmp_path), "rabi", "a", 3)
    result = preprocessor.process(raw, "test_meta")

    assert result.raw_data_path == raw.path
    assert result.metadata["source"] == "tool4_pass_through"
    assert result.metadata["raw_data_path"] == raw.path


def test_process_auto_generates_id(tmp_path):
    preprocessor = DataPreprocessor(str(Path(tmp_path) / "processed"))
    raw = _make_raw_data(str(tmp_path), "t1", "b", 3)
    result = preprocessor.process(raw)

    assert Path(result.path).parent.name.startswith("t1_b")


def test_process_t1_passthrough(tmp_path):
    preprocessor = DataPreprocessor(str(Path(tmp_path) / "processed"))
    raw = _make_raw_data(str(tmp_path), "t1", "b", 5)
    result = preprocessor.process(raw)

    # T1: values should decay
    assert result.data[0] > result.data[-1]


def test_process_output_dir_created(tmp_path):
    output_dir = str(Path(tmp_path) / "nonexistent" / "processed")
    preprocessor = DataPreprocessor(output_dir)
    raw = _make_raw_data(str(tmp_path), "rabi", "a", 2)
    result = preprocessor.process(raw)

    assert Path(result.path).exists()
