"""Tool5 — 分析层：对预处理数据拟合物理模型，生成图表和分析报告。

分析流程:
    1. 根据实验类型选择拟合模型 (rabi/t1/ramsey)
    2. scipy.curve_fit 非线性最小二乘拟合 (含参数边界)
    3. 计算拟合质量 (R², RMSE)
    4. 绘制拟合图 → fit.png
    5. 可选: LLM 生成自然语言报告
    6. 保存分析结果 → analysis_result.json
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

from ..models.analysis_result import AnalysisResult
from ..models.processed_data import ProcessedData
from ..utils.fitting_models import rabi_model, t1_model, ramsey_model
from ..utils.plotting import plot_fit

# 实验类型 → (拟合函数, 参数名列表)
_MODEL_REGISTRY = {
    "rabi": (rabi_model, ["A", "T_rabi", "phi", "y0"]),
    "t1": (t1_model, ["A", "T1", "y0"]),
    "ramsey": (ramsey_model, ["A", "T2_star", "f_detune", "phi", "y0"]),
}

# 拟合初始猜测 (p0)，基于典型物理参数
_INITIAL_GUESS = {
    "rabi": [1.0, 1000.0, 0.0, 0.0],       # A, T_rabi(ns), phi, y0
    "t1": [1.0, 5000.0, 0.0],               # A, T1(ns), y0
    "ramsey": [0.5, 30000.0, 0.0005, 0.0, 0.5],  # A, T2*(ns), f(GHz), phi, y0
}

# 参数边界 (lower, upper)，防止非物理解 (如负的衰减时间)
_BOUNDS = {
    "rabi": ([0.0, 0.0, -np.pi, -1.0], [2.0, np.inf, np.pi, 1.0]),
    "t1": ([0.0, 0.0, -1.0], [2.0, np.inf, 1.0]),
    "ramsey": ([0.0, 0.0, 0.0, -np.pi, -1.0], [2.0, np.inf, np.inf, np.pi, 1.0]),
}


class ExperimentAnalyzer:
    """实验数据分析器：拟合 + 绘图 + 报告生成。

    Args:
        llm_client: LLM 客户端 (可选)。传入时生成自然语言报告，为 None 时跳过。
        output_dir: 分析结果输出目录。
    """

    def __init__(self, llm_client=None, output_dir: str = "data/analysis"):
        self.llm = llm_client
        self.output_dir = Path(output_dir)

    def analyze(self, processed: ProcessedData) -> AnalysisResult:
        """执行完整的拟合分析管线。

        Args:
            processed: Tool4 输出的预处理数据

        Returns:
            AnalysisResult 包含拟合参数、误差、图表路径和 LLM 报告
        """
        exp_type = processed.experiment_type
        if exp_type not in _MODEL_REGISTRY:
            raise ValueError(f"不支持的实验类型: '{exp_type}'")

        model_fn, param_names = _MODEL_REGISTRY[exp_type]
        x = processed.scan_values.astype(np.float64)
        y = processed.data.astype(np.float64)

        # 非线性最小二乘拟合
        p0 = _INITIAL_GUESS[exp_type]
        bounds = _BOUNDS[exp_type]
        popt, pcov = curve_fit(model_fn, x, y, p0=p0, bounds=bounds, maxfev=10000)

        # 参数误差 (协方差矩阵对角线平方根)
        perr = np.sqrt(np.diag(pcov))
        parameters = dict(zip(param_names, popt))
        parameter_errors = dict(zip(param_names, perr))

        # 拟合质量指标
        y_fit = model_fn(x, *popt)
        ss_res = np.sum((y - y_fit) ** 2)          # 残差平方和
        ss_tot = np.sum((y - np.mean(y)) ** 2)     # 总平方和
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = np.sqrt(ss_res / len(y))            # 均方根误差

        fit_quality = {"r_squared": float(r_squared), "rmse": float(rmse)}

        # 输出目录
        eid = f"{exp_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir = self.output_dir / eid
        out_dir.mkdir(parents=True, exist_ok=True)

        # 绘图 (散点 + 平滑拟合曲线)
        plot_path = str(out_dir / "fit.png")
        plot_fit(x, y, y_fit, exp_type, plot_path, model_fn, popt)

        # LLM 报告 (可选)
        llm_summary = ""
        if self.llm is not None:
            llm_summary = self._generate_report(exp_type, parameters, fit_quality)

        # 保存结果
        result = AnalysisResult(
            experiment_id=eid,
            experiment_type=exp_type,
            parameters=parameters,
            parameter_errors=parameter_errors,
            fit_quality=fit_quality,
            result_path=str(out_dir / "analysis_result.json"),
            plot_path=plot_path,
            llm_summary=llm_summary,
        )
        self._save_result(result)
        return result

    def _generate_report(self, exp_type: str, params: dict, quality: dict) -> str:
        """调用 LLM 根据拟合结果生成简短的自然语言报告。"""
        prompt = f"""请根据以下量子测控实验结果生成一段简短报告（2-3句）：
实验类型：{exp_type}
拟合参数：{json.dumps(params, ensure_ascii=False)}
拟合优度 R²：{quality.get('r_squared', 0):.3f}
如果拟合质量差，请指出可能原因。"""
        try:
            return self.llm.complete("你是量子测控实验分析助手。", prompt)
        except Exception:
            return ""

    def _save_result(self, result: AnalysisResult) -> None:
        """将 AnalysisResult 序列化为 JSON 文件。"""
        data = {
            "experiment_id": result.experiment_id,
            "experiment_type": result.experiment_type,
            "parameters": result.parameters,
            "parameter_errors": result.parameter_errors,
            "fit_quality": result.fit_quality,
            "llm_summary": result.llm_summary,
        }
        p = Path(result.result_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
