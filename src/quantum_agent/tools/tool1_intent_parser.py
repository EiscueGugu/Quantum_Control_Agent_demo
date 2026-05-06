"""Tool1 — 实验定义层：将自然语言指令转换为结构化的 ExperimentIR。

管线: 用户输入 → System Prompt + LLM → JSON 解析 → 归一化 → 校验 → ExperimentIR

核心类:
    LLMClient (ABC)        — LLM 调用抽象接口
    DeepSeekClient         — DeepSeek API 实现 (兼容 OpenAI SDK)
    MockLLMClient          — 测试用 Mock
    ExperimentIRParser     — 解析器主类
    ClarificationNeeded    — 信息不完整时返回的澄清请求
"""

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..models.experiment_ir import ExperimentIR, Pulse, ScanParameter


@dataclass
class ClarificationNeeded:
    """当 LLM 输出信息不完整时返回此对象，提示用户补充。

    用户补充信息后可通过 continue_with_clarification() 继续解析。
    """
    message: str                                  # 给用户的提示消息
    questions: List[str]                          # 具体需要回答的问题列表
    partial_ir: Dict[str, Any] = field(default_factory=dict)  # 已解析的部分结果


class LLMClient(ABC):
    """LLM 调用抽象基类，隔离真实 API 和 Mock 实现。"""

    @abstractmethod
    def complete(self, system_prompt: str, user_message: str) -> str: ...


class DeepSeekClient(LLMClient):
    """DeepSeek API 客户端，兼容 OpenAI SDK 协议。

    API Key 从环境变量 DEEPSEEK_API_KEY 读取，或以参数传入。
    使用 temperature=0 确保输出稳定可复现。
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        self.model = model
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
        )

    def complete(self, system_prompt: str, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content


class MockLLMClient(LLMClient):
    """测试用 Mock 客户端，忽略输入直接返回预设字符串。"""

    def __init__(self, response: str):
        self.response = response

    def complete(self, system_prompt: str, user_message: str) -> str:
        return self.response


# ═══════════════════════════════════════════════════════════════
# System Prompt — 指导 LLM 生成标准 JSON 实验描述
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一个量子测控实验助手。根据用户的自然语言指令，生成一个表示实验的 JSON 对象。

必须遵循以下结构：
{
  "experiment_type": "rabi" | "t1" | "ramsey",
  "qubit": "a" | "b" | ...,
  "fixed_params": {
    "power_dBm": 10.0,
    "readout_freq_MHz": 5400.0,
    ...
  },
  "scan": {
    "name": "pulse_duration",
    "unit": "us",
    "start": 0.0,
    "stop": 100.0,
    "num_points": 100,
    "linear": true
  },
  "pulse": [
    { "name": "x180", "shape": "gaussian", "duration_ns": "$pulse_duration", "amplitude": 1.0 },
    { "name": "readout", "shape": "square", "duration_ns": 1000, "amplitude": 1.0 }
  ]
}

脉冲命名规则（必须严格遵循，不得使用别名）：
- x180 = π脉冲（180度旋转脉冲）
- x90  = π/2脉冲（90度旋转脉冲）
- x90y = Y轴90度脉冲
- readout = 读出脉冲
- wait = 等待/自由演化时间
禁止使用 "pi"、"π"、"pi/2"、"π/2"、"free_evolution" 等别名。只使用以上标准名称。

扫描参数命名规则（必须严格遵循）：
- rabi 实验：扫描参数名必须为 "pulse_duration"
- t1 实验：扫描参数名必须为 "wait_time"
- ramsey 实验：扫描参数名必须为 "free_evolution_time"

重要规则：
1. 如果用户给出"步长×点数"，则 stop = step * (num_points - 1) 或 step * num_points，从0开始。
2. 脉冲序列必须符合实验类型：
   - rabi: [x180(时长扫描), readout]
   - t1: [x180(固定时长), wait(扫描时长), readout]
   - ramsey: [x90, wait(扫描时长), x90, readout]
3. 扫描参数名必须与脉冲序列中的占位符一致（如 $pulse_duration）。
4. 未指定的固定参数可以省略或设为 null，随后填充默认值。
5. 如果用户未指定单位，默认使用 us。
6. 只输出 JSON，不要有其他解释文字。
7. **关键**: 如果用户没有明确给出扫描的起始值(start)、终止值(stop)或点数(num_points)，必须将对应字段设为 null，绝对不要自行推测或编造数值。如果用户没有指定量子比特(qubit)，qubit 字段也必须设为 null。例如用户只说"做 T1 实验"而未给出扫描范围，scan 中的 start/stop/num_points 和 qubit 都必须为 null。
8. 固定时长脉冲的默认值（用户未指定时使用）：x180 默认 duration_ns=50，x90 默认 duration_ns=25，readout 默认 duration_ns=1000。

现在请根据用户指令生成 JSON："""


# ═══════════════════════════════════════════════════════════════
# 校验常量 & 别名映射
# ═══════════════════════════════════════════════════════════════

REQUIRED_EXPERIMENT_FIELDS = ["experiment_type", "qubit"]
REQUIRED_SCAN_FIELDS = ["name", "unit", "start", "stop", "num_points"]
REQUIRED_PULSE_FIELDS = ["name", "shape", "duration_ns", "amplitude"]
VALID_EXPERIMENT_TYPES = {"rabi", "t1", "ramsey"}
VALID_PULSE_NAMES = {"x180", "x90", "x90y", "readout", "wait"}
VALID_SHAPES = {"gaussian", "square"}
VALID_UNITS = {"us", "ns", "ms", "s"}

# 脉冲名别名 → 标准名 (LLM 可能输出非标准名称，在此归一化)
PULSE_NAME_ALIASES = {
    "pi": "x180", "π": "x180",
    "pi/2": "x90", "π/2": "x90", "pi_half": "x90",
    "free_evolution": "wait", "delay": "wait",
}

# 扫描参数名别名 → 标准名
SCAN_NAME_ALIASES = {
    "wait_duration": "wait_time",
    "evolution_time": "free_evolution_time", "tau": "free_evolution_time",
    "pulse_width": "pulse_duration", "duration": "pulse_duration",
}


# ═══════════════════════════════════════════════════════════════
# ExperimentIRParser — 解析器主类
# ═══════════════════════════════════════════════════════════════

class ExperimentIRParser:
    """将自然语言解析为 ExperimentIR 的完整管线。

    流程:
        1. 拼接 System Prompt + 用户输入 → 调用 LLM
        2. 从 LLM 回复中提取 JSON (支持 ```json 代码块)
        3. json.loads 反序列化
        4. _normalize() 别名归一化
        5. _build_ir() 校验 + 构建 ExperimentIR
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def parse(self, user_input: str) -> ExperimentIR | ClarificationNeeded:
        """解析用户的自然语言指令。

        Args:
            user_input: 自然语言实验描述

        Returns:
            ExperimentIR (成功) 或 ClarificationNeeded (需要补充信息)
        """
        response = self.llm.complete(SYSTEM_PROMPT, user_input)
        json_str = self._extract_json(response)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return ClarificationNeeded(
                message=f"LLM 返回了无法解析的 JSON：{e}",
                questions=["请重新以 JSON 格式描述实验定义。"],
                partial_ir={"raw_response": response},
            )
        return self._build_ir(data)

    def continue_with_clarification(
        self, partial_ir: Dict[str, Any], answers: Dict[str, Any]
    ) -> ExperimentIR | ClarificationNeeded:
        """用户补充信息后继续解析。

        Args:
            partial_ir: 上次返回的 ClarificationNeeded.partial_ir
            answers: 用户补充的字段，如 {"qubit": "a"}
        """
        merged = {**partial_ir}
        for key, value in answers.items():
            if key == "fixed_params" and isinstance(value, dict):
                merged.setdefault("fixed_params", {}).update(value)
            elif key == "scan" and isinstance(value, dict):
                merged.setdefault("scan", {}).update(value)
            elif key == "pulse" and isinstance(value, list):
                merged["pulse"] = value
            else:
                merged[key] = value
        return self._build_ir(merged)

    def _extract_json(self, text: str) -> str:
        """从 LLM 回复中提取 JSON 字符串。

        处理三种情况:
            1. ```json ... ```  (Markdown 代码块)
            2. { ... }          (裸 JSON)
            3. 其他              (兜底返回原文)
        """
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            return match.group(1)
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return match.group(0)
        return text

    def _normalize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """归一化 LLM 输出: 将别名转换为标准名称。

        处理内容:
            - 脉冲名: pi → x180, pi/2 → x90 等
            - 扫描参数名: wait_duration → wait_time 等
            - 同步更新脉冲中的 $占位符
        """
        normalized = dict(data)

        scan = normalized.get("scan")
        if isinstance(scan, dict) and scan.get("name"):
            scan_name = scan["name"]
            if scan_name in SCAN_NAME_ALIASES:
                new_name = SCAN_NAME_ALIASES[scan_name]
                # 同步更新脉冲中的占位符
                pulses = normalized.get("pulse", [])
                for p in pulses:
                    dur = p.get("duration_ns")
                    if isinstance(dur, str) and dur == f"${scan_name}":
                        p["duration_ns"] = f"${new_name}"
                scan["name"] = new_name

        pulses = normalized.get("pulse", [])
        for p in pulses:
            name = p.get("name")
            if isinstance(name, str) and name in PULSE_NAME_ALIASES:
                p["name"] = PULSE_NAME_ALIASES[name]

        return normalized

    def _build_ir(self, data: Dict[str, Any]) -> ExperimentIR | ClarificationNeeded:
        """校验归一化后的数据并构建 ExperimentIR。

        校验步骤:
            1. 必填实验字段 (experiment_type, qubit)
            2. 实验类型白名单
            3. 扫描参数必填字段 + 合法单位
            4. 脉冲序列非空 + 各脉冲必填字段 + 合法名称/形状
        """
        data = self._normalize(data)
        questions: List[str] = []

        # 校验实验字段
        for field in REQUIRED_EXPERIMENT_FIELDS:
            if field not in data or data[field] is None:
                questions.append(f"请指定实验的 {field} 字段。")

        exp_type = data.get("experiment_type")
        if exp_type and exp_type not in VALID_EXPERIMENT_TYPES:
            questions.append(
                f"不支持的实验类型 '{exp_type}'，可选：{', '.join(sorted(VALID_EXPERIMENT_TYPES))}。"
            )

        # 校验扫描参数
        scan_data = data.get("scan") or {}
        for field in REQUIRED_SCAN_FIELDS:
            if field not in scan_data or scan_data[field] is None:
                questions.append(f"请指定扫描参数的 {field} 字段。")
        if scan_data.get("unit") and scan_data["unit"] not in VALID_UNITS:
            questions.append(
                f"不支持的单位 '{scan_data['unit']}'，可选：{', '.join(sorted(VALID_UNITS))}。"
            )

        # 校验脉冲序列
        pulses: List[Dict[str, Any]] = data.get("pulse", [])
        if not pulses:
            questions.append("脉冲序列不能为空，请至少指定一个脉冲。")
        else:
            for i, p in enumerate(pulses):
                for field in REQUIRED_PULSE_FIELDS:
                    if field not in p or p[field] is None:
                        questions.append(f"脉冲 [{i}] 缺少 {field} 字段。")
                if p.get("name") and p["name"] not in VALID_PULSE_NAMES:
                    questions.append(
                        f"脉冲 [{i}] 的名称 '{p['name']}' 不在标准列表中：{', '.join(sorted(VALID_PULSE_NAMES))}。"
                    )
                if p.get("shape") and p["shape"] not in VALID_SHAPES:
                    questions.append(
                        f"脉冲 [{i}] 的形状 '{p['shape']}' 不支持，可选：{', '.join(sorted(VALID_SHAPES))}。"
                    )

        if questions:
            return ClarificationNeeded(
                message="实验描述信息不完整，请补充：",
                questions=questions,
                partial_ir=data,
            )

        # 构建 ScanParameter
        scan = ScanParameter(
            name=scan_data["name"],
            unit=scan_data["unit"],
            start=float(scan_data["start"]),
            stop=float(scan_data["stop"]),
            num_points=int(scan_data["num_points"]),
            linear=scan_data.get("linear", True),
        )

        # 构建 Pulse 列表 (保留 $占位符)
        pulse_objs = []
        for p in pulses:
            dur = p["duration_ns"]
            if isinstance(dur, str) and dur.startswith("$"):
                duration_ns: float | str = dur   # 扫描占位符，保留字符串
            else:
                duration_ns = float(dur)          # 固定数值
            pulse_objs.append(
                Pulse(
                    name=p["name"],
                    shape=p["shape"],
                    duration_ns=duration_ns,
                    amplitude=float(p["amplitude"]),
                )
            )

        return ExperimentIR(
            experiment_type=data["experiment_type"],
            qubit=data["qubit"],
            fixed_params=data.get("fixed_params") or {},
            scan=scan,
            pulse=pulse_objs,
        )
