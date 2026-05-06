"""Agent 创建测试。使用 create_agent() 工厂函数。"""

from unittest.mock import patch

from quantum_agent.agent_core import create_agent


def test_create_agent_sets_api_key_error():
    """无 DEEPSEEK_API_KEY 时应抛出错误。"""
    try:
        with patch.dict("os.environ", {}, clear=True):
            create_agent()
            assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "DEEPSEEK_API_KEY" in str(e)


def test_create_agent_returns_callable():
    """有 API key 时 create_agent 应返回 LangChain Agent。"""
    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
        with patch("quantum_agent.agent_core.ChatOpenAI"):
            agent = create_agent()
            assert agent is not None
            # LangChain agent should be a CompiledStateGraph
            assert hasattr(agent, "invoke")
