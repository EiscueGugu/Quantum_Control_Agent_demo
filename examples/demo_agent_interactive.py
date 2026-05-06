"""
Agent 交互模式演示 — 流式输出

用户输入自然语言目标，Agent 自主决策调用工具链完成实验。
每一步工具调用和结果实时显示，最终回答在流程完成后输出。
输入 quit 退出。
"""

import os
import sys

# 必须在所有第三方库导入前抑制 HuggingFace 日志
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TQDM_DISABLE"] = "1"

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantum_agent import create_agent


def main():
    if "DEEPSEEK_API_KEY" not in os.environ:
        print("错误: 未找到 DEEPSEEK_API_KEY，请检查 .env 文件。")
        sys.exit(1)

    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "instruments.yaml")
    agent = create_agent(config_path=config_path)

    print("=" * 60)
    print("  量子测控 Agent (带记忆)")
    print("  输入实验目标，Agent 自主完成全流程：如对比特Q1进行拉比振荡实验。")
    print("  输入 quit 退出")
    print("=" * 60)
    print()

    # 会话级 thread_id，Agent 在同一会话中记住历史
    import uuid
    thread_id = uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("> ").strip()
        if user_input.lower() == "quit":
            print("再见！")
            break
        if not user_input:
            continue

        step = 0
        final_answer = ""

        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="updates",
            config=config,
        ):
            for node_name, node_output in chunk.items():
                if node_name == "model":
                    msgs = node_output.get("messages", [])
                    if not msgs:
                        continue
                    last_msg = msgs[-1]

                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            step += 1
                            args = tc.get("args", {})
                            s = f"  [{step}] {tc['name']}("
                            if args:
                                arg_str = str(args)
                                if len(arg_str) > 60:
                                    arg_str = arg_str[:60] + "..."
                                s += arg_str
                            s += ")"
                            print(s)
                    elif hasattr(last_msg, "content") and last_msg.content:
                        final_answer = str(last_msg.content)

                elif node_name == "tools":
                    for tool_msg in node_output.get("messages", []):
                        lines = str(tool_msg.content).split("\n")
                        first = lines[0]
                        if len(first) > 100:
                            first = first[:100] + "..."
                        print(f"        -> {first}")
                        for line in lines[1:8]:
                            stripped = line.strip()
                            if stripped:
                                print(f"           {stripped}")

        print(f"\n{'=' * 60}")
        print(final_answer)
        print(f"{'=' * 60}")
        print()


if __name__ == "__main__":
    main()
