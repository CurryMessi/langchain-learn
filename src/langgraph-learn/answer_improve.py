from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Annotated
import operator

class ImprovementState(TypedDict):
    question: str
    answer: str
    quality_score: float
    iteration: Annotated[int, operator.add]
    max_iterations: int

def generate_answer(state: ImprovementState):
    """生成或改进答案"""
    iteration = state.get("iteration", 0)
    
    # 模拟答案逐步改进
    if iteration == 0:
        answer = "简单回答：这是一个基础答案。"
        score = 0.5
    elif iteration == 1:
        answer = "改进回答：这是一个更详细的答案，包含更多信息。"
        score = 0.7
    else:
        answer = "完善回答：经过仔细思考的完整答案，包含背景和建议。"
        score = 0.9
    
    print(f"第{iteration + 1}次生成 (质量: {score})")
    
    return {
        "answer": answer,
        "quality_score": score,
        "iteration": 1  # 每次+1（因为有 reducer）
    }

def check_quality(state):
    """检查质量，不更新状态"""
    return state

def should_continue(state: ImprovementState) -> str:
    """决定是继续还是结束"""
    if state["quality_score"] >= 0.8:
        return "done"
    elif state["iteration"] >= state["max_iterations"]:
        return "done"
    else:
        return "improve"

# 构建带循环的图
workflow = StateGraph(ImprovementState)

workflow.add_node("generate", generate_answer)
workflow.add_node("check", check_quality)

workflow.add_edge(START, "generate")
workflow.add_edge("generate", "check")

# 关键：条件边可以循环回去
workflow.add_conditional_edges(
    "check",
    should_continue,
    {
        "improve": "generate",  # 循环
        "done": END             # 结束
    }
)

app = workflow.compile()

result = app.invoke({
    "question": "什么是人工智能？",
    "answer": "",
    "quality_score": 0.0,
    "iteration": 0,
    "max_iterations": 5
})

print(f"\n最终答案: {result['answer']}")
print(f"最终质量: {result['quality_score']}")
print(f"迭代次数: {result['iteration']}")