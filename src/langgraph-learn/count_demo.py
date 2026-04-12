from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# 1. 定义状态结构
class CounterState(TypedDict):
    count: int
    history: list[str]

# 2. 定义节点函数
def increment_node(state: CounterState):
    """计数器 +1"""
    new_count = state["count"] + 1
    message = f"count: {state['count']} → {new_count}"
    
    return {
        "count": new_count,
        "history": state["history"] + [message]
    }

def double_node(state: CounterState):
    """计数器翻倍"""
    new_count = state["count"] * 2
    message = f"count: {state['count']} → {new_count}"
    
    return {
        "count": new_count,
        "history": state["history"] + [message]
    }

def report_node(state: CounterState):
    """报告结果"""
    return {"history": state["history"] + [f"最终结果: {state['count']}"]}

# 3. 创建图
workflow = StateGraph[CounterState, None, CounterState, CounterState](CounterState)

# 4. 添加节点
workflow.add_node("increment", increment_node)
workflow.add_node("double", double_node)
workflow.add_node("report", report_node)

# 5. 连接节点
workflow.add_edge(START, "increment")
workflow.add_edge("increment", "double")
workflow.add_edge("double", "report")
workflow.add_edge("report", END)

# 6. 编译
app = workflow.compile()

# 7. 运行
result = app.invoke({
    "count": 5,
    "history": ["开始执行"]
})

print("执行历史:")
for step in result["history"]:
    print(f"  - {step}")
print(f"\n最终计数: {result['count']}")