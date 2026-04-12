from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


@tool
def add(a: float, b: float) -> float:
    """两数相加"""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """两数相乘"""
    return a * b

load_dotenv()

# 创建模型
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
    max_tokens=100,
)

# 创建 ReAct Agent
agent = create_agent(
    model=llm,
    tools=[add, multiply],
    system_prompt="你是一个数学计算助手。"
)

# 使用
result = agent.invoke({
    "messages": [{"role": "user", "content": "计算 (10 + 5) × 3"}]
})
print(result["messages"][-1].content)