"""基于 LangChain create_agent 的量子测控 Agent。

使用 LangChain 内置 ReAct 循环，工具通过 @tool 装饰器注册。

用法:
    agent = create_agent(model, tools=tools, system_prompt=...)
    result = agent.invoke({"messages": [{"role": "user", "content": goal}]})
"""

import logging
import os

# 在导入 HuggingFace 库之前抑制日志和进度条
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
for name in ["sentence_transformers", "transformers", "huggingface_hub", "tqdm"]:
    logging.getLogger(name).setLevel(logging.ERROR)

from langchain.agents import create_agent as lc_create_agent
from langchain.tools import tool as lc_tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from .models.experiment_ir import ExperimentIR
from .retrieval.retriever import RAGRetriever
from .tools.tool1_intent_parser import ClarificationNeeded, ExperimentIRParser
from .tools.tool2_compiler import ExperimentCompiler
from .tools.tool3_executor import ExperimentExecutor
from .tools.tool4_preprocessor import DataPreprocessor
from .tools.tool5_analyzer import ExperimentAnalyzer


# ═══════════════════════════════════════════════════════════
# Agent 内部状态
# ═══════════════════════════════════════════════════════════

class _AgentState:
    """实验流水线中的共享状态。"""
    def __init__(self):
        self.ir: ExperimentIR | None = None
        self.compiled = None
        self.raw = None
        self.processed = None
        self.result = None


# ═══════════════════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════════════════

def create_agent(
    model_name: str = "deepseek-chat",
    config_path: str = "config/instruments.yaml",
):
    """创建基于 LangChain ReAct 循环的量子测控 Agent。

    Args:
        model_name: DeepSeek 模型名
        config_path: instruments.yaml 路径
        verbose: 是否打印中间步骤

    Returns:
        CompiledStateGraph (LangChain Agent)，用 .invoke() 调用

    Usage:
        agent = create_agent()
        result = agent.invoke({
            "messages": [{"role": "user", "content": "对 qubit a 做 Rabi 实验 0-3us 扫 30 点，并分析结果"}]
        })
        print(result["messages"][-1].content)
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")

    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0,
    )

    # 内部组件
    parser = ExperimentIRParser(None)  # 后面会重新绑定
    compiler = ExperimentCompiler(config_path)
    executor = ExperimentExecutor(config_path)
    preprocessor = DataPreprocessor()
    analyzer = ExperimentAnalyzer(None)

    state = _AgentState()

    # RAG 检索器
    retriever = RAGRetriever("data/chroma_db_v2")

    # 重新绑定 parser 和 analyzer 的 LLM
    parser.llm = _ChatModelAdapter(model)

    # ── 工具定义 ─────────────────────────────────────────

    @lc_tool
    def define_experiment(instruction: str) -> str:
        """将自然语言指令转换为实验定义（实验类型、扫描参数、脉冲序列）。
        必须在所有实验流程中首先调用。参数 instruction 为完整的自然语言实验描述。"""
        result = parser.parse(instruction)
        if isinstance(result, ClarificationNeeded):
            questions = "\n".join(f"  - {q}" for q in result.questions)
            return f"信息不足，需要补充:\n{questions}"
        state.ir = result
        return (
            f"实验定义成功。类型={result.experiment_type}, qubit={result.qubit}, "
            f"扫描={result.scan.name}[{result.scan.start}..{result.scan.stop}{result.scan.unit}]"
            f"x{result.scan.num_points}, 脉冲数={len(result.pulse)}"
        )

    @lc_tool
    def compile_experiment() -> str:
        """编译实验定义为 VSG/DAQ/Trigger 的底层执行文件（波形 + 配置）。
        必须在 define_experiment 成功后调用。无需参数。"""
        ir = state.ir
        if ir is None:
            return "错误: 没有实验定义。请先调用 define_experiment。"
        compiled = compiler.compile(ir)
        state.compiled = compiled
        return (
            f"实验编译完成。扫描点数={len(compiled.scan_values)}, "
            f"总采样数={compiled.total_samples}, 输出目录={compiled.work_dir}"
        )

    @lc_tool
    def execute_experiment() -> str:
        """驱动虚拟仪器执行实验，采集原始数据。必须在 compile_experiment 成功后调用。无需参数。"""
        compiled = state.compiled
        if compiled is None:
            return "错误: 没有编译结果。请先调用 compile_experiment。"
        raw = executor.execute(compiled)
        state.raw = raw
        return f"仪器加载执行文件...正在执行实验...数据采集完成。数据点数={len(raw.data)}, 信号范围=[{raw.data.min():.4f}, {raw.data.max():.4f}]"

    @lc_tool
    def preprocess_data() -> str:
        """预处理原始数据（当前为透传）。必须在 execute_experiment 成功后调用。无需参数。"""
        raw = state.raw
        if raw is None:
            return "错误: 没有原始数据。请先调用 execute_experiment。"
        processed = preprocessor.process(raw)
        state.processed = processed
        return f"预处理完成（透传）。数据点数={len(processed.data)}"

    @lc_tool
    def analyze_experiment() -> str:
        """对预处理数据拟合物理模型，生成分析报告和图表。必须在 preprocess_data 成功后调用。无需参数。"""
        processed = state.processed
        if processed is None:
            return "错误: 没有预处理数据。请先调用 preprocess_data。"
        result = analyzer.analyze(processed)
        state.result = result
        lines = [f"分析完成。拟合参数:"]
        for k, v in result.parameters.items():
            lines.append(f"  {k} = {v:.4f} ± {result.parameter_errors[k]:.4f}")
        lines.append(
            f"拟合质量: R²={result.fit_quality['r_squared']:.4f}, "
            f"RMSE={result.fit_quality['rmse']:.4f}"
        )
        lines.append(f"拟合图表: {result.plot_path}")
        return "\n".join(lines)

    @lc_tool
    def ask_user(question: str) -> str:
        """当实验信息不足时向用户提问。参数 question 为具体问题。"""
        return f"[需要用户回答] {question}"

    @lc_tool
    def search_knowledge(query: str) -> str:
        """搜索量子测控知识库，获取实验原理、参数建议、故障排查等信息。
        当用户询问"为什么"、"怎么办"、"什么原因"等理论或方法问题时使用。
        参数 query 为搜索关键词或问题。"""
        results = retriever.search(query, k=3)
        if not results or results[0].startswith("(知识库"):
            return "知识库为空或未初始化。"
        return "\n---\n".join(results)

    tools = [
        define_experiment, compile_experiment, execute_experiment,
        preprocess_data, analyze_experiment, search_knowledge, ask_user,
    ]

    system_prompt = """你是一个量子测控实验助手。使用提供的工具完成实验任务。

流水线顺序 (必须严格遵守):
  1. define_experiment → 2. compile_experiment → 3. execute_experiment
  → 4. preprocess_data → 5. analyze_experiment

规则:
- "做实验并分析" = 执行全部 1→2→3→4→5，一步不能少
- 每步成功后立即自动调用下一个工具，不要等待、不要确认
- analyze_experiment 完成后立即给出最终总结，在此之前绝不可给出总结
- define_experiment 返回"信息不足"时，使用 ask_user 提问
- compile/execute/preprocess/analyze 无需参数，直接调用
- 用户询问理论知识、实验原理、故障排查、参数建议时，先调用 search_knowledge 检索知识库再回答
- 用中文回复，最终总结必须引用拟合参数的具体数值和拟合图表/结果文件的路径"""

    import sqlite3
    conn = sqlite3.connect("data/agent_memory.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    return lc_create_agent(
        model, tools=tools, system_prompt=system_prompt, checkpointer=checkpointer,
    )


# ═══════════════════════════════════════════════════════════
# LLMClient 适配器 (让 LangChain ChatModel 兼容旧接口)
# ═══════════════════════════════════════════════════════════

class _ChatModelAdapter:
    """将 LangChain ChatModel 适配为 LLMClient 接口，供 ExperimentIRParser 使用。"""

    def __init__(self, chat_model):
        self._model = chat_model

    def complete(self, system_prompt: str, user_message: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = self._model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])
        return response.content
