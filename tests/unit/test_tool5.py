import json
from pathlib import Path

import numpy as np
import pytest

from quantum_agent.models.analysis_result import AnalysisResult
from quantum_agent.models.processed_data import ProcessedData
from quantum_agent.tools.tool5_analyzer import ExperimentAnalyzer
from quantum_agent.utils.fitting_models import rabi_model, t1_model, ramsey_model


# --- Helpers ---

def _make_processed(tmp_path: str, exp_type: str, n: int = 20) -> ProcessedData:
    rng = np.random.default_rng(42)

    if exp_type == "rabi":
        x = np.linspace(0, 3000, n)
        true_params = [0.5, 1000.0, 0.0, 0.0]
        y = rabi_model(x, *true_params) + rng.normal(0, 0.03, n)
    elif exp_type == "t1":
        x = np.linspace(0, 10000, n)
        true_params = [1.0, 5000.0, 0.0]
        y = t1_model(x, *true_params) + rng.normal(0, 0.03, n)
    elif exp_type == "ramsey":
        x = np.linspace(0, 10000, n)
        true_params = [0.5, 5000.0, 0.002, 0.0, 0.5]
        y = ramsey_model(x, *true_params) + rng.normal(0, 0.03, n)
    else:
        raise ValueError(f"Unknown: {exp_type}")

    return ProcessedData(
        path=str(Path(tmp_path) / "proc.h5"),
        experiment_type=exp_type,
        qubit="a",
        scan_values=x,
        data=y,
        raw_data_path=str(Path(tmp_path) / "raw.h5"),
    )


# --- Fitting model tests ---

def test_rabi_model_shape():
    x = np.linspace(0, 2000, 100)
    y = rabi_model(x, 1.0, 1000.0, 0.0, 0.0)
    assert len(y) == 100
    assert y.min() >= -0.1
    assert y.max() <= 1.1


def test_t1_model_decays():
    x = np.array([0, 1000, 5000])
    y = t1_model(x, 1.0, 2000.0, 0.0)
    assert y[0] > y[1] > y[2]


def test_ramsey_model_oscillates():
    x = np.linspace(0, 10000, 500)
    y = ramsey_model(x, 0.5, 5000.0, 0.002, 0.0, 0.5)
    diffs = np.diff(y)
    assert np.any(diffs > 0) and np.any(diffs < 0)


# --- Analyzer tests ---

def test_analyze_rabi(tmp_path):
    analyzer = ExperimentAnalyzer(output_dir=str(Path(tmp_path) / "analysis"))
    proc = _make_processed(str(tmp_path), "rabi")
    result = analyzer.analyze(proc)

    assert isinstance(result, AnalysisResult)
    assert result.experiment_type == "rabi"
    assert "T_rabi" in result.parameters
    assert result.fit_quality["r_squared"] > 0.5
    assert Path(result.plot_path).exists()
    assert Path(result.result_path).exists()


def test_analyze_t1(tmp_path):
    analyzer = ExperimentAnalyzer(output_dir=str(Path(tmp_path) / "analysis"))
    proc = _make_processed(str(tmp_path), "t1")
    result = analyzer.analyze(proc)

    assert result.experiment_type == "t1"
    assert "T1" in result.parameters
    assert result.parameters["T1"] > 0
    assert result.fit_quality["r_squared"] > 0.5


def test_analyze_ramsey(tmp_path):
    analyzer = ExperimentAnalyzer(output_dir=str(Path(tmp_path) / "analysis"))
    proc = _make_processed(str(tmp_path), "ramsey")
    result = analyzer.analyze(proc)

    assert result.experiment_type == "ramsey"
    assert "T2_star" in result.parameters
    assert "f_detune" in result.parameters


def test_analyze_unknown_type_raises(tmp_path):
    analyzer = ExperimentAnalyzer(output_dir=str(Path(tmp_path) / "analysis"))
    proc = _make_processed(str(tmp_path), "rabi")
    proc.experiment_type = "unknown"
    with pytest.raises(ValueError, match="不支持"):
        analyzer.analyze(proc)


def test_analyze_saves_result_json(tmp_path):
    analyzer = ExperimentAnalyzer(output_dir=str(Path(tmp_path) / "analysis"))
    proc = _make_processed(str(tmp_path), "t1")
    result = analyzer.analyze(proc)

    with open(result.result_path, "r") as f:
        data = json.load(f)
    assert data["experiment_type"] == "t1"
    assert "parameters" in data
    assert "fit_quality" in data


def test_analyze_parameter_errors(tmp_path):
    analyzer = ExperimentAnalyzer(output_dir=str(Path(tmp_path) / "analysis"))
    proc = _make_processed(str(tmp_path), "rabi")
    result = analyzer.analyze(proc)

    assert len(result.parameter_errors) == len(result.parameters)
    for err in result.parameter_errors.values():
        assert err >= 0


def test_analyze_plot_is_png(tmp_path):
    analyzer = ExperimentAnalyzer(output_dir=str(Path(tmp_path) / "analysis"))
    proc = _make_processed(str(tmp_path), "ramsey")
    result = analyzer.analyze(proc)

    assert result.plot_path.endswith(".png")
    assert Path(result.plot_path).exists()
    assert Path(result.plot_path).stat().st_size > 0
