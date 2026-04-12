from langchain_core.tools import tool
from typing import Optional
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """
    获取城市的天气信息
    
    Args:
        city: 城市名称，如"北京"、"上海"
    """
    # 模拟天气数据（实际项目中调用天气 API）
    weather_data = {
        "北京": "晴天，25°C",
        "上海": "多云，22°C",
        "深圳": "小雨，28°C",
    }
    return weather_data.get(city, f"{city}的天气暂时无法获取")


@tool
def search_database(query: str, category: Optional[str] = None) -> str:
    """
    在数据库中搜索信息
    
    Args:
        query: 搜索关键词
        category: 可选的分类过滤器，如"科技"、"健康"、"教育"
    """
    database = {
        "科技": ["人工智能正在改变世界", "5G技术的应用"],
        "健康": ["健康饮食的重要性", "运动与长寿的关系"],
        "教育": ["在线学习的趋势", "终身学习的价值"]
    }
    
    results = []
    
    if category and category in database:
        for item in database[category]:
            if query.lower() in item.lower():
                results.append(f"[{category}] {item}")
    else:
        for cat, items in database.items():
            for item in items:
                if query.lower() in item.lower():
                    results.append(f"[{cat}] {item}")
    
    if results:
        return "找到以下结果:\n" + "\n".join(results)
    else:
        return f"没有找到关于 '{query}' 的结果"

@tool
def calculator(expression: str) -> str:
    """执行数学计算"""
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except:
        return "计算错误"

@tool  
def search_knowledge(query: str) -> str:
    """搜索知识库"""
    knowledge = {
        "LangChain": "LangChain是一个用于开发LLM应用的框架，支持工具、代理、内存管理等功能。",
        "机器学习": "机器学习是AI的子集，让系统能从数据中自动学习和改进。",
    }
    for key, value in knowledge.items():
        if key in query:
            return value
    return f"未找到关于'{query}'的信息"


def main() -> None:
    result = get_weather.invoke({"city": "北京"})
    print(result)

    print(f"工具名称: {get_weather.name}")
    print(f"工具描述: {get_weather.description}")
    print(f"参数结构: {get_weather.args}")

    tools = [get_weather, search_database]

    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.7,
        max_tokens=100,
    )

    llm.bind_tools(tools)

    response = llm.invoke("今天北京天气怎么样？")

    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"模型想调用: {tool_call['name']}")
            print(f"传入参数: {tool_call['args']}")

    agent = create_agent(
        model=llm,
        tools=[get_weather, calculator, search_knowledge],
        system_prompt="你是一个专业的中文助手。仔细分析用户问题，选择合适的工具来回答。"
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": "北京今天天气怎么样？"}]
    })

    final_message = result["messages"][-1]
    print(f"回答: {final_message.content}")


if __name__ == "__main__":
    main()
