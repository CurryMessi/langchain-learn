from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

# 1. 定义状态
class SupervisorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str  # 下一个执行的 Agent

# 2. 定义各种工具
@tool
def check_system_status(component: str) -> str:
    """检查系统组件状态"""
    status = {"database": "正常", "server": "负载高", "api": "正常"}
    return status.get(component, "未知")

@tool
def get_product_info(product_name: str) -> str:
    """获取产品信息"""
    products = {
        "基础版": "99元/月，基础功能",
        "专业版": "299元/月，高级功能",
    }
    return products.get(product_name, "产品不存在")

@tool
def query_invoice(order_id: str) -> str:
    """查询发票"""
    invoices = {"ORD001": "已开具", "ORD002": "处理中"}
    return invoices.get(order_id, "订单不存在")

# 3. 定义专业 Agent 节点
def tech_agent_node(state):
    """技术支持 Agent"""
    last_message = state["messages"][-1].content

    if "系统" in last_message or "服务器" in last_message:
        result = check_system_status.invoke({"component": "server"})
        response = f"【技术支持】系统状态: {result}"
    else:
        response = "【技术支持】请描述您遇到的技术问题。"

    return {"messages": [AIMessage(content=response)]}

def sales_agent_node(state):
    """销售 Agent"""
    last_message = state["messages"][-1].content

    for product in ["基础版", "专业版"]:
        if product in last_message:
            info = get_product_info.invoke({"product_name": product})
            return {"messages": [AIMessage(content=f"【销售顾问】{info}")]}

    return {"messages": [AIMessage(content="【销售顾问】我们有基础版和专业版，需要了解哪个？")]}

def billing_agent_node(state):
    """账单 Agent"""
    return {"messages": [AIMessage(content="【账单专员】请提供订单号，我来查询。")]}

# 4. Supervisor 节点
def supervisor_node(state):
    """Supervisor：决定下一步"""
    last_message = state["messages"][-1].content

    tech_keywords = ["错误", "bug", "崩溃", "系统", "服务器"]
    sales_keywords = ["价格", "购买", "产品", "套餐"]
    billing_keywords = ["发票", "支付", "退款", "账单"]

    if any(kw in last_message for kw in tech_keywords):
        next_agent = "tech_support"
    elif any(kw in last_message for kw in sales_keywords):
        next_agent = "sales"
    elif any(kw in last_message for kw in billing_keywords):
        next_agent = "billing"
    else:
        next_agent = "sales"  # 默认销售

    return {"next": next_agent}

# 5. 路由函数
def route_after_supervisor(state) -> Literal["tech_support", "sales", "billing"]:
    return state["next"]

# 6. 构建图
workflow = StateGraph(SupervisorState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("tech_support", tech_agent_node)
workflow.add_node("sales", sales_agent_node)
workflow.add_node("billing", billing_agent_node)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {
        "tech_support": "tech_support",
        "sales": "sales",
        "billing": "billing",
    }
)
workflow.add_edge("tech_support", END)
workflow.add_edge("sales", END)
workflow.add_edge("billing", END)

app = workflow.compile()

# 7. 测试
test_cases = [
    "我遇到500错误，系统无法访问",
    "专业版多少钱？",
    "我需要查发票",
]

for query in test_cases:
    print(f"用户: {query}")
    result = app.invoke({"messages": [HumanMessage(content=query)], "next": ""})
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(msg.content)
    print()