import json
import os
import tempfile
from pathlib import Path

import pytest

from quantum_agent.models.experiment_ir import ExperimentIR, Pulse, ScanParameter
from quantum_agent.tools.tool1_intent_parser import (
    ClarificationNeeded,
    DeepSeekClient,
    ExperimentIRParser,
    MockLLMClient,
)
from quantum_agent.utils.file_io import load_experiment_ir, save_experiment_ir


def _json_response(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


# --- Fixtures ---

@pytest.fixture
def rabi_json():
    return _json_response({
        "experiment_type": "rabi",
        "qubit": "a",
        "fixed_params": {"power_dBm": 10.0, "readout_freq_MHz": 5400.0},
        "scan": {"name": "pulse_duration", "unit": "us", "start": 0.0, "stop": 100.0, "num_points": 101, "linear": True},
        "pulse": [
            {"name": "x180", "shape": "gaussian", "duration_ns": "$pulse_duration", "amplitude": 1.0},
            {"name": "readout", "shape": "square", "duration_ns": 1000, "amplitude": 1.0},
        ],
    })


@pytest.fixture
def t1_json():
    return _json_response({
        "experiment_type": "t1",
        "qubit": "b",
        "fixed_params": {},
        "scan": {"name": "wait_time", "unit": "us", "start": 0.0, "stop": 50.0, "num_points": 101, "linear": True},
        "pulse": [
            {"name": "x180", "shape": "gaussian", "duration_ns": 50, "amplitude": 1.0},
            {"name": "wait", "shape": "square", "duration_ns": "$wait_time", "amplitude": 0.0},
            {"name": "readout", "shape": "square", "duration_ns": 1000, "amplitude": 1.0},
        ],
    })


@pytest.fixture
def ramsey_json():
    return _json_response({
        "experiment_type": "ramsey",
        "qubit": "a",
        "fixed_params": {},
        "scan": {"name": "free_evolution_time", "unit": "us", "start": 0.0, "stop": 10.0, "num_points": 200, "linear": True},
        "pulse": [
            {"name": "x90", "shape": "gaussian", "duration_ns": 25, "amplitude": 1.0},
            {"name": "wait", "shape": "square", "duration_ns": "$free_evolution_time", "amplitude": 0.0},
            {"name": "x90", "shape": "gaussian", "duration_ns": 25, "amplitude": 1.0},
            {"name": "readout", "shape": "square", "duration_ns": 1000, "amplitude": 1.0},
        ],
    })


# --- Successful parse tests ---

def test_parse_rabi_success(rabi_json):
    parser = ExperimentIRParser(MockLLMClient(rabi_json))
    result = parser.parse("对 qubit a 做 Rabi 实验，脉冲时长 0 到 100 us")
    assert isinstance(result, ExperimentIR)
    assert result.experiment_type == "rabi"
    assert result.qubit == "a"
    assert result.scan.name == "pulse_duration"
    assert result.scan.unit == "us"
    assert result.scan.start == 0.0
    assert result.scan.stop == 100.0
    assert result.scan.num_points == 101
    assert result.scan.linear is True
    assert result.fixed_params == {"power_dBm": 10.0, "readout_freq_MHz": 5400.0}
    assert len(result.pulse) == 2
    assert result.pulse[0].name == "x180"
    assert result.pulse[0].duration_ns == "$pulse_duration"
    assert result.pulse[1].name == "readout"
    assert result.pulse[1].duration_ns == 1000


def test_parse_t1_success(t1_json):
    parser = ExperimentIRParser(MockLLMClient(t1_json))
    result = parser.parse("对 qubit b 做 T1 实验，等待 0 到 50 us")
    assert isinstance(result, ExperimentIR)
    assert result.experiment_type == "t1"
    assert result.qubit == "b"
    assert result.scan.name == "wait_time"
    assert result.scan.num_points == 101
    assert len(result.pulse) == 3
    assert result.pulse[1].name == "wait"
    assert result.pulse[1].amplitude == 0.0


def test_parse_ramsey_success(ramsey_json):
    parser = ExperimentIRParser(MockLLMClient(ramsey_json))
    result = parser.parse("对 qubit a 做 Ramsey 实验")
    assert isinstance(result, ExperimentIR)
    assert result.experiment_type == "ramsey"
    assert result.scan.name == "free_evolution_time"
    assert len(result.pulse) == 4
    assert result.pulse[0].name == "x90"
    assert result.pulse[2].name == "x90"


# --- JSON extraction tests ---

def test_json_in_codeblock():
    wrapped = '```json\n{"experiment_type":"rabi","qubit":"a","scan":{"name":"x","unit":"us","start":0,"stop":10,"num_points":10},"pulse":[{"name":"x180","shape":"gaussian","duration_ns":"$x","amplitude":1.0}]}```'
    parser = ExperimentIRParser(MockLLMClient(wrapped))
    result = parser.parse("anything")
    assert isinstance(result, ExperimentIR)


def test_json_without_codeblock():
    raw = '{"experiment_type":"t1","qubit":"b","scan":{"name":"t","unit":"us","start":0,"stop":5,"num_points":5},"pulse":[{"name":"x180","shape":"gaussian","duration_ns":50,"amplitude":1.0}]}'
    parser = ExperimentIRParser(MockLLMClient(raw))
    result = parser.parse("anything")
    assert isinstance(result, ExperimentIR)


# --- Clarification tests ---

def test_missing_qubit_asks_clarification():
    bad_json = _json_response({
        "experiment_type": "rabi",
        "scan": {"name": "pulse_duration", "unit": "us", "start": 0, "stop": 10, "num_points": 10},
        "pulse": [{"name": "x180", "shape": "gaussian", "duration_ns": 50, "amplitude": 1.0}],
    })
    parser = ExperimentIRParser(MockLLMClient(bad_json))
    result = parser.parse("做 Rabi 实验")
    assert isinstance(result, ClarificationNeeded)
    assert any("qubit" in q.lower() for q in result.questions)
    assert result.partial_ir["experiment_type"] == "rabi"


def test_missing_scan_fields_asks_clarification():
    bad_json = _json_response({
        "experiment_type": "t1",
        "qubit": "a",
        "scan": {"name": "wait_time"},
        "pulse": [{"name": "x180", "shape": "gaussian", "duration_ns": 50, "amplitude": 1.0}],
    })
    parser = ExperimentIRParser(MockLLMClient(bad_json))
    result = parser.parse("做 T1 实验")
    assert isinstance(result, ClarificationNeeded)
    assert len(result.questions) > 0


def test_empty_pulse_asks_clarification():
    bad_json = _json_response({
        "experiment_type": "ramsey",
        "qubit": "a",
        "scan": {"name": "t", "unit": "us", "start": 0, "stop": 10, "num_points": 10},
        "pulse": [],
    })
    parser = ExperimentIRParser(MockLLMClient(bad_json))
    result = parser.parse("做 Ramsey")
    assert isinstance(result, ClarificationNeeded)
    assert any("脉冲" in q for q in result.questions)


def test_invalid_experiment_type_asks_clarification():
    bad_json = _json_response({
        "experiment_type": "spin_echo",
        "qubit": "a",
        "scan": {"name": "t", "unit": "us", "start": 0, "stop": 10, "num_points": 10},
        "pulse": [{"name": "x180", "shape": "gaussian", "duration_ns": 50, "amplitude": 1.0}],
    })
    parser = ExperimentIRParser(MockLLMClient(bad_json))
    result = parser.parse("做 spin echo")
    assert isinstance(result, ClarificationNeeded)
    assert any("spin_echo" in q for q in result.questions)


def test_invalid_json_asks_clarification():
    parser = ExperimentIRParser(MockLLMClient("not valid json at all"))
    result = parser.parse("whatever")
    assert isinstance(result, ClarificationNeeded)
    assert "raw_response" in result.partial_ir


def test_invalid_pulse_name_asks_clarification():
    bad_json = _json_response({
        "experiment_type": "rabi",
        "qubit": "a",
        "scan": {"name": "t", "unit": "us", "start": 0, "stop": 10, "num_points": 10},
        "pulse": [{"name": "unknown_pulse", "shape": "gaussian", "duration_ns": 50, "amplitude": 1.0}],
    })
    parser = ExperimentIRParser(MockLLMClient(bad_json))
    result = parser.parse("test")
    assert isinstance(result, ClarificationNeeded)
    assert any("unknown_pulse" in q for q in result.questions)


# --- Clarification follow-up test ---

def test_clarification_then_reparse():
    bad_json = _json_response({
        "experiment_type": "rabi",
        "scan": {"name": "pulse_duration", "unit": "us", "start": 0, "stop": 10, "num_points": 10},
        "pulse": [{"name": "x180", "shape": "gaussian", "duration_ns": 50, "amplitude": 1.0}],
    })
    parser = ExperimentIRParser(MockLLMClient(bad_json))
    result1 = parser.parse("做 Rabi")
    assert isinstance(result1, ClarificationNeeded)

    result2 = parser.continue_with_clarification(result1.partial_ir, {"qubit": "a"})
    assert isinstance(result2, ExperimentIR)
    assert result2.experiment_type == "rabi"
    assert result2.qubit == "a"


# --- Save / Load round-trip ---

def test_save_and_load_ir():
    ir = ExperimentIR(
        experiment_type="rabi",
        qubit="a",
        fixed_params={"power_dBm": 10.0},
        scan=ScanParameter(name="pulse_duration", unit="us", start=0.0, stop=100.0, num_points=101),
        pulse=[
            Pulse(name="x180", shape="gaussian", duration_ns="$pulse_duration", amplitude=1.0),
            Pulse(name="readout", shape="square", duration_ns=1000, amplitude=1.0),
        ],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_ir.json"
        save_experiment_ir(ir, str(path))
        loaded = load_experiment_ir(str(path))
        assert loaded.experiment_type == ir.experiment_type
        assert loaded.qubit == ir.qubit
        assert loaded.scan.name == ir.scan.name
        assert loaded.scan.num_points == ir.scan.num_points
        assert loaded.pulse[0].duration_ns == "$pulse_duration"
        assert loaded.pulse[0].amplitude == 1.0


# --- DeepSeekClient ---

def test_deepseek_client_no_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        DeepSeekClient()


# --- Normalization tests ---

def test_normalize_pi_to_x180():
    json_with_alias = _json_response({
        "experiment_type": "rabi",
        "qubit": "a",
        "scan": {"name": "pulse_duration", "unit": "us", "start": 0, "stop": 10, "num_points": 10},
        "pulse": [
            {"name": "pi", "shape": "gaussian", "duration_ns": 50, "amplitude": 1.0},
            {"name": "readout", "shape": "square", "duration_ns": 1000, "amplitude": 1.0},
        ],
    })
    parser = ExperimentIRParser(MockLLMClient(json_with_alias))
    result = parser.parse("做 Rabi")
    assert isinstance(result, ExperimentIR)
    assert result.pulse[0].name == "x180"


def test_normalize_pi_half_to_x90():
    json_with_alias = _json_response({
        "experiment_type": "ramsey",
        "qubit": "a",
        "scan": {"name": "free_evolution_time", "unit": "us", "start": 0, "stop": 10, "num_points": 10},
        "pulse": [
            {"name": "pi/2", "shape": "gaussian", "duration_ns": 25, "amplitude": 1.0},
            {"name": "wait", "shape": "square", "duration_ns": "$free_evolution_time", "amplitude": 0.0},
            {"name": "pi/2", "shape": "gaussian", "duration_ns": 25, "amplitude": 1.0},
            {"name": "readout", "shape": "square", "duration_ns": 1000, "amplitude": 1.0},
        ],
    })
    parser = ExperimentIRParser(MockLLMClient(json_with_alias))
    result = parser.parse("做 Ramsey")
    assert isinstance(result, ExperimentIR)
    assert result.pulse[0].name == "x90"
    assert result.pulse[2].name == "x90"


def test_normalize_free_evolution_pulse_to_wait():
    json_with_alias = _json_response({
        "experiment_type": "t1",
        "qubit": "a",
        "scan": {"name": "wait_time", "unit": "us", "start": 0, "stop": 10, "num_points": 10},
        "pulse": [
            {"name": "x180", "shape": "gaussian", "duration_ns": 50, "amplitude": 1.0},
            {"name": "free_evolution", "shape": "square", "duration_ns": "$wait_time", "amplitude": 0.0},
            {"name": "readout", "shape": "square", "duration_ns": 1000, "amplitude": 1.0},
        ],
    })
    parser = ExperimentIRParser(MockLLMClient(json_with_alias))
    result = parser.parse("做 T1")
    assert isinstance(result, ExperimentIR)
    assert result.pulse[1].name == "wait"


def test_normalize_scan_name_and_placeholder():
    json_with_alias = _json_response({
        "experiment_type": "t1",
        "qubit": "a",
        "scan": {"name": "wait_duration", "unit": "us", "start": 0, "stop": 10, "num_points": 10},
        "pulse": [
            {"name": "x180", "shape": "gaussian", "duration_ns": 50, "amplitude": 1.0},
            {"name": "wait", "shape": "square", "duration_ns": "$wait_duration", "amplitude": 0.0},
            {"name": "readout", "shape": "square", "duration_ns": 1000, "amplitude": 1.0},
        ],
    })
    parser = ExperimentIRParser(MockLLMClient(json_with_alias))
    result = parser.parse("做 T1")
    assert isinstance(result, ExperimentIR)
    assert result.scan.name == "wait_time"
    assert result.pulse[1].duration_ns == "$wait_time"


def test_unknown_pulse_still_asks_clarification():
    json_bad = _json_response({
        "experiment_type": "rabi",
        "qubit": "a",
        "scan": {"name": "pulse_duration", "unit": "us", "start": 0, "stop": 10, "num_points": 10},
        "pulse": [{"name": "completely_unknown", "shape": "gaussian", "duration_ns": 50, "amplitude": 1.0}],
    })
    parser = ExperimentIRParser(MockLLMClient(json_bad))
    result = parser.parse("test")
    assert isinstance(result, ClarificationNeeded)
