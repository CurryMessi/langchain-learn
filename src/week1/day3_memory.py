"""
LangChain Day 3 实践：记忆管理深度探索

适配 LangChain 1.x 的消息历史写法：
1. 使用 `prompt | model` 构建链
2. 使用 `InMemoryChatMessageHistory` 保存历史
3. 使用 `RunnableWithMessageHistory` 管理会话级记忆
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


load_dotenv()


def check_deepseek_config() -> bool:
    """检查 DeepSeek API 配置。"""
    print("🔍 检查 DeepSeek 配置...")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("   ❌ DEEPSEEK_API_KEY 未设置")
        print("   请设置环境变量或创建 .env 文件")
        return False

    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    print(f"   ✅ API 密钥已设置: {masked_key}")
    return True


def build_deepseek_model(*, temperature: float = 0.7, max_tokens: int | None = None) -> ChatOpenAI:
    """使用 OpenAI 兼容接口连接 DeepSeek。"""
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=2,
    )


def build_prompt(system_message: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )


def extract_text(message: object) -> str:
    """兼容字符串和分块内容。"""
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def format_history(messages: Iterable[BaseMessage]) -> str:
    """把消息历史格式化成便于观察的文本。"""
    lines: List[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "用户"
        elif isinstance(message, AIMessage):
            role = "AI"
        else:
            role = message.__class__.__name__
        lines.append(f"{role}: {extract_text(message)}")
    return "\n".join(lines) if lines else "（暂无历史）"


def get_recent_turn_messages(messages: List[BaseMessage], turns: int) -> List[BaseMessage]:
    """保留最近 N 轮对话。每轮通常包含 1 条用户消息和 1 条 AI 消息。"""
    if turns <= 0:
        return []
    return messages[-2 * turns :]


def build_history_aware_chain(
    system_message: str,
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> RunnableWithMessageHistory:
    """构建带会话历史的链。"""
    model = build_deepseek_model(temperature=temperature, max_tokens=max_tokens)
    chain = build_prompt(system_message) | model
    store: Dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def demo_buffer_memory() -> None:
    """演示完整保留历史的做法。"""
    print("\n" + "=" * 50)
    print("演示1: 完整消息历史（替代旧版 ConversationBufferMemory）")
    print("=" * 50)

    try:
        conversation = build_history_aware_chain("你是一个友好且记性很好的 AI 助手。")
        config = {"configurable": {"session_id": "buffer_demo"}}

        print("💬 先进行两轮对话:")
        for user_input in ["你好，我叫张三", "我喜欢打篮球"]:
            response = conversation.invoke({"input": user_input}, config=config)
            print(f"   你: {user_input}")
            print(f"   AI: {extract_text(response)}")

        history = conversation.get_session_history("buffer_demo")
        print("\n📝 当前完整历史:")
        print(format_history(history.messages))

        response = conversation.invoke({"input": "我的名字是什么？"}, config=config)
        print(f"\n   你: 我的名字是什么？")
        print(f"   AI: {extract_text(response)}")

        print("\n📝 更新后的完整历史:")
        print(format_history(history.messages))

    except Exception as e:
        print(f"   ❌ 演示失败: {e}")


def demo_window_memory() -> None:
    """演示最新写法中的窗口记忆思路。"""
    print("\n" + "=" * 50)
    print("演示2: 窗口记忆（替代旧版 ConversationBufferWindowMemory）")
    print("=" * 50)

    try:
        history = InMemoryChatMessageHistory()
        conversations = [
            ("你好，我叫张三", "你好，张三！"),
            ("我来自北京", "北京是个很美的城市！"),
            ("我喜欢编程", "编程很有趣！"),
            ("我今年 25 岁", "25 岁很棒，正是学习和成长的好阶段！"),
        ]

        window_turns = 2
        print(f"📊 模拟对话过程（只查看最近 {window_turns} 轮）:")

        for index, (user_input, ai_output) in enumerate(conversations, start=1):
            history.add_user_message(user_input)
            history.add_ai_message(ai_output)

            recent_messages = get_recent_turn_messages(history.messages, window_turns)
            print(f"\n   第 {index} 轮后:")
            print(format_history(recent_messages))

        print("\n💡 在 LangChain 1.x 中，窗口记忆通常由开发者自己裁剪消息列表。")

    except Exception as e:
        print(f"   ❌ 演示失败: {e}")


def demo_chat_with_memory() -> None:
    """演示手动控制窗口的完整多轮对话系统。"""
    print("\n" + "=" * 50)
    print("演示3: 完整多轮对话系统（手动控制最近 3 轮上下文）")
    print("=" * 50)

    try:
        model = build_deepseek_model(temperature=0.7, max_tokens=150)
        prompt = build_prompt("你是一个友好、专业，且善于记住上下文的 AI 助手。")
        chain = prompt | model
        history = InMemoryChatMessageHistory()
        window_turns = 3

        print("🤖 开始多轮对话（仅把最近 3 轮发给模型）")
        print("-" * 30)

        test_dialogue = [
            "你好，我是小王，来自上海",
            "我是一名软件工程师",
            "我刚才说我是做什么工作的？",
        ]

        for user_input in test_dialogue:
            recent_messages = get_recent_turn_messages(history.messages, window_turns)
            response = chain.invoke(
                {
                    "chat_history": recent_messages,
                    "input": user_input,
                }
            )
            answer = extract_text(response)

            history.add_user_message(user_input)
            history.add_ai_message(answer)

            print(f"   你: {user_input}")
            print(f"   AI: {answer}")
            print()

        print("📝 当前完整历史:")
        print(format_history(history.messages))
        print("\n✅ 对话完成！这就是 LangChain 1.x 中更常见的记忆管理方式。")

    except Exception as e:
        print(f"   ❌ 演示失败: {e}")


def demo_runnable_with_message_history() -> None:
    """演示官方推荐的会话级记忆管理。"""
    print("\n" + "=" * 50)
    print("演示4: RunnableWithMessageHistory - 官方推荐方式")
    print("=" * 50)

    try:
        conversation = build_history_aware_chain("你是一个友好且专业的 AI 助手。")

        session_a = {"configurable": {"session_id": "user_zhangsan"}}
        session_b = {"configurable": {"session_id": "user_lisi"}}

        print("💬 测试会话隔离:")

        response1 = conversation.invoke({"input": "你好，我叫李四"}, config=session_a)
        print("   Session A -> 你: 你好，我叫李四")
        print(f"   Session A -> AI: {extract_text(response1)}")

        response2 = conversation.invoke({"input": "我的名字是什么？"}, config=session_a)
        print("\n   Session A -> 你: 我的名字是什么？")
        print(f"   Session A -> AI: {extract_text(response2)}")

        response3 = conversation.invoke({"input": "你记得我叫什么吗？"}, config=session_b)
        print("\n   Session B -> 你: 你记得我叫什么吗？")
        print(f"   Session B -> AI: {extract_text(response3)}")

        print("\n✅ 演示完成！不同 session_id 会拥有各自独立的记忆。")

    except Exception as e:
        print(f"   ❌ 演示失败: {e}")


def main() -> None:
    """主函数：协调所有演示。"""
    print("=" * 60)
    print("LangChain Day 3 实践：记忆管理深度探索")
    print("=" * 60)

    if not check_deepseek_config():
        print("\n⚠️ 配置未就绪，请先设置 DEEPSEEK_API_KEY")
        print("   参考 Day 1 的环境配置指南")
        return

    demo_buffer_memory()
    demo_window_memory()
    demo_chat_with_memory()
    demo_runnable_with_message_history()

    print("\n" + "=" * 60)
    print("📋 今日学习总结")
    print("=" * 60)
    print("1. ✅ 理解了消息历史在记忆管理中的核心作用")
    print("2. ✅ 掌握了完整历史与窗口历史两种常见方案")
    print("3. ✅ 实践了 DeepSeek 环境下的 LangChain 1.x 写法")
    print("4. ✅ 体验了 RunnableWithMessageHistory 的会话隔离能力")
    print("\n🔮 下一步：在 Day 4 的实战演练中应用这些模式")
    print("   构建更复杂的对话系统和智能代理")


if __name__ == "__main__":
    main()
