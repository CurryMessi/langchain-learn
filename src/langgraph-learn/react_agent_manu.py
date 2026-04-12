from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated, Literal
from typing_extensions import TypedDict
import operator
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool
import os

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
    max_tokens=100,
)
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iterations: Annotated[int, operator.add]

# 工具定义...
@tool
def python_executor(code: str) -> str:
    """执行 Python 代码"""
    return exec(code)

@tool
def text_analyzer(text: str) -> str:
    """分析文本"""
    return text


tools = [python_executor, text_analyzer]
tool_node = ToolNode(tools)

# 模型绑定工具
llm_with_tools = llm.bind_tools(tools)

def call_model(state: AgentState):
    """调用模型决策"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response], "iterations": 1}

def should_continue(state) -> Literal["tools", "__end__"]:
    """判断是否需要调用工具"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "__end__": END
})
workflow.add_edge("tools", "agent")

app = workflow.compile()