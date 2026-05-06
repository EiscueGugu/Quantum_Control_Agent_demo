"""量子测控 Agent 主入口。

提供两种使用模式:
    1. Agent 模式 (推荐): agent.run("自然语言目标") → LLM 自主完成全流程
    2. 手动模式 (兼容):  agent.define_experiment() → compile_experiment() → ...
"""

from .agent_core import create_agent
from .models.analysis_result import AnalysisResult
from .models.compiled_experiment import CompiledExperiment
from .models.experiment_ir import ExperimentIR
from .models.processed_data import ProcessedData
from .models.raw_data import RawData
from .tools.tool1_intent_parser import (
    ClarificationNeeded,
    ExperimentIRParser,
    LLMClient,
)
from .tools.tool2_compiler import ExperimentCompiler
from .tools.tool3_executor import ExperimentExecutor
from .tools.tool4_preprocessor import DataPreprocessor
from .tools.tool5_analyzer import ExperimentAnalyzer


class QuantumControlAgent:
    """量子测控 Agent，支持 Agent 模式和手动模式。

    Agent 模式 (使用 LangChain create_agent):
        agent = QuantumControlAgent(DeepSeekClient())
        result = agent.run("对 qubit a 做 Rabi 实验 0-3us 扫 30 点，并分析结果")

    手动模式 (兼容):
        ir = agent.define_experiment("对 qubit a 做 Rabi 实验...")
        compiled = agent.compile_experiment(ir)
        ...
    """

    def __init__(self, llm_client: LLMClient, config_path: str = "config/instruments.yaml"):
        self._lc_agent = create_agent(config_path=config_path)

        self.tool1 = ExperimentIRParser(llm_client)
        self.tool2 = ExperimentCompiler(config_path)
        self.tool3 = ExperimentExecutor(config_path)
        self.tool4 = DataPreprocessor()
        self.tool5 = ExperimentAnalyzer(llm_client)
        self._pending_clarification: ClarificationNeeded | None = None

    # ── Agent 模式 ────────────────────────────────────────────

    def run(self, goal: str) -> str:
        """Agent 模式: LLM 自主决定调用工具完成实验目标。

        Args:
            goal: 自然语言实验目标

        Returns:
            LLM 生成的实验结果总结
        """
        result = self._lc_agent.invoke({
            "messages": [{"role": "user", "content": goal}]
        })
        return result["messages"][-1].content

    # ── Tool1: 实验定义 ──────────────────────────────────────

    def define_experiment(self, user_input: str) -> ExperimentIR | ClarificationNeeded:
        result = self.tool1.parse(user_input)
        if isinstance(result, ClarificationNeeded):
            self._pending_clarification = result
        else:
            self._pending_clarification = None
        return result

    def clarify(self, answers: dict) -> ExperimentIR | ClarificationNeeded:
        if self._pending_clarification is None:
            return ClarificationNeeded(
                message="没有待澄清的实验定义。请先调用 define_experiment。",
                questions=[],
            )
        result = self.tool1.continue_with_clarification(
            self._pending_clarification.partial_ir, answers
        )
        if isinstance(result, ClarificationNeeded):
            self._pending_clarification = result
        else:
            self._pending_clarification = None
        return result

    @property
    def has_pending_clarification(self) -> bool:
        return self._pending_clarification is not None

    @property
    def pending_clarification(self) -> ClarificationNeeded | None:
        return self._pending_clarification

    # ── Tool2-5: 手动模式 ──────────────────────────────────

    def compile_experiment(self, ir: ExperimentIR) -> CompiledExperiment:
        return self.tool2.compile(ir)

    def execute_experiment(self, compiled: CompiledExperiment) -> RawData:
        return self.tool3.execute(compiled)

    def preprocess_data(self, raw: RawData) -> ProcessedData:
        return self.tool4.process(raw)

    def analyze_experiment(self, processed: ProcessedData) -> AnalysisResult:
        return self.tool5.analyze(processed)
