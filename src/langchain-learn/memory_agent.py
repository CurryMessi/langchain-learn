from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from tools import get_weather, calculator, search_knowledge

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
    max_tokens=100,
)

# 创建记忆存储
checkpointer = MemorySaver()

# 创建带记忆的 Agent
agent = create_agent(
    model=llm,
    tools=[get_weather, calculator, search_knowledge],
    system_prompt="你是一个专业助手，能记住之前的对话。",
    checkpointer=checkpointer  # 启用记忆
)

# 会话配置（同一个 thread_id 共享记忆）
config = {"configurable": {"thread_id": "session_001"}}

# 第一轮对话
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫小明"}]},
    config
)
print(result1["messages"][-1].content)

# 第二轮对话（Agent 会记得用户叫小明）
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    config
)
print(result2["messages"][-1].content)  # 会回答"小明"


# 流式调用
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "北京天气如何？"}]},
    config,
    stream_mode="values"
):
    messages = chunk.get("messages")
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, "content") and last_message.content:
            print(last_message.content, end="", flush=True)
print()  # 换行