"""Tool5 输出 — 拟合分析结果。"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class AnalysisResult:
    """物理拟合分析结果。

    包含: 拟合参数及误差、拟合质量指标 (R², RMSE)、图表路径、可选的 LLM 报告。
    """
    experiment_id: str                    # 实验 ID
    experiment_type: str                  # "rabi" / "t1" / "ramsey"
    parameters: Dict[str, float] = field(default_factory=dict)           # 拟合参数值
    parameter_errors: Dict[str, float] = field(default_factory=dict)     # 参数标准误差
    fit_quality: Dict[str, float] = field(default_factory=dict)          # {"r_squared": ..., "rmse": ...}
    result_path: str = ""                 # analysis_result.json 路径
    plot_path: str = ""                   # fit.png 路径
    llm_summary: str = ""                 # LLM 生成的自然语言报告
