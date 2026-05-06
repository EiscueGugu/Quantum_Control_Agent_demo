# Quantum Control Agent

一个基于 LLM 的智能量子测控实验助手。将自然语言指令自动转化为完整的实验流程——从实验定义、波形编译、虚拟仪器执行、数据采集，到物理模型拟合分析，全程由 AI Agent 自主决策完成。

## 项目介绍

- **自然语言驱动** — 说"对 qubit a 做 Rabi 实验 0-3us 扫 30 点"，Agent 自动完成全部 5 步流水线
- **ReAct 智能体** — 基于 LangChain `create_agent`，LLM 自主决定何时调用工具、何时追问用户
- **5 工具流水线** — define → compile → execute → preprocess → analyze
- **虚拟仪器** — VirtualVSG / VirtualDAQ / VirtualTrigger 模拟真实硬件
- **物理模拟** — Rabi 振荡、T1 弛豫、Ramsey 干涉的仿真数据生成
- **scipy 拟合** — 自动拟合物理模型，计算 R²/RMSE，生成拟合图
- **RAG 知识检索** — 内置 Chroma 向量库，检索实验原理与故障排查知识
- **流式输出** — 工具调用步骤实时可见
- **持久记忆** — SqliteSaver 跨对话记忆
- **LLM 分析报告** — DeepSeek-V4 生成自然语言实验总结

## 快速开始

### 环境要求

- Python 3.10+
- DeepSeek API Key

### 安装

```bash
git clone https://github.com/yourname/quantum_control_agent.git
cd quantum_control_agent
pip install -e ".[dev]"
```

### 配置

复制环境变量模板并填入你的 API Key：

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-xxx
```

### 构建 RAG 索引（可选）

```bash
python -c "from quantum_agent.retrieval import RAGIndexer; RAGIndexer().build('knowledge_base')"
```

### 运行

```bash
# Agent 交互模式（带 RAG + 记忆）
python examples/demo_agent_interactive.py
```

## 架构

```
用户自然语言
    │
    ▼
┌──────────────────────────────────────┐
│  Quantum Agent (LangChain ReAct)      │
│  ┌─────────────────────────────────┐ │
│  │ LLM (DeepSeek-V4)               │ │
│  │ 思考 → 选择工具 → 观察结果 → ... │ │
│  └──────────────┬──────────────────┘ │
│                 │                     │
│  ┌──────────────▼──────────────────┐ │
│  │ Tool Set                        │ │
│  │ define → compile → execute      │ │
│  │ → preprocess → analyze          │ │
│  │ + search_knowledge (RAG)        │ │
│  │ + ask_user                      │ │
│  └─────────────────────────────────┘ │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  虚拟仪器层                          │
│  VirtualVSG  VirtualDAQ  VirtualTrigger│
└──────────────────────────────────────┘
```

## 项目结构

```
quantum_control_agent/
├── src/quantum_agent/
│   ├── agent_core.py              # Agent 核心 (create_agent)
│   ├── agent.py                   # 兼容层
│   ├── tools/                     # 5 个 Tool
│   │   ├── tool1_intent_parser.py   # 自然语言 → ExperimentIR
│   │   ├── tool2_compiler.py        # ExperimentIR → 波形文件
│   │   ├── tool3_executor.py        # 虚拟仪器执行
│   │   ├── tool4_preprocessor.py    # 预处理（透传）
│   │   └── tool5_analyzer.py        # 拟合分析
│   ├── instruments/               # 虚拟仪器
│   │   ├── vsg.py / daq.py / trigger.py
│   ├── models/                    # 数据模型
│   ├── retrieval/                 # RAG 模块
│   └── utils/                     # 拟合模型 + 绘图
├── tests/unit/                    # 单元测试 (73)
├── examples/                      # 演示脚本
├── knowledge_base/                # RAG 知识库
├── config/instruments.yaml        # 仪器配置
├── docs/                          # API 参考文档
└── data/                          # 运行时数据 (git ignored)
```

## 运行测试

```bash
pytest tests/ -v
```

## License

MIT
