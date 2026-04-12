"""
一个真实的 LangChain 多智能体示例。

实现思路：
1. 使用 `langchain.agents.create_agent` 创建多个子 agent
2. 把子 agent 包装成 tools
3. 再用一个 supervisor agent 统一调度这些 tools

这比手写 if/else 流程更接近官方推荐的多 agent 模式。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT_DIR / ".env"


def load_project_env() -> None:
    """显式加载项目根目录的 .env，避免 stdin 脚本场景下 find_dotenv 的问题。"""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)


def build_model() -> ChatOpenAI:
    """优先使用 DeepSeek，其次回退到 OpenAI。"""
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if deepseek_api_key:
        return ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            api_key=deepseek_api_key,
            temperature=0.2,
            max_retries=2,
        )

    if openai_api_key:
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            api_key=openai_api_key,
            temperature=0.2,
            max_retries=2,
        )

    raise RuntimeError(
        "未找到可用的 API Key。请在 .env 中设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY。"
    )


def extract_text(agent_result: Dict[str, object]) -> str:
    """从 agent.invoke(...) 的结果中提取最后一条消息文本。"""
    message = agent_result["messages"][-1]
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def run_subagent(agent_name: str, agent, task: str) -> str:
    """执行单个子 agent，并把输出打印出来，方便学习观察。"""
    print(f"\n{'=' * 16} {agent_name} {'=' * 16}")
    print(f"任务: {task}\n")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": task,
                }
            ]
        }
    )
    answer = extract_text(result)
    print(answer)
    return answer


def build_team():
    """构建 supervisor 和它要调用的多个子 agent。"""
    model = build_model()

    product_manager = create_agent(
        model=model,
        name="product_manager",
        system_prompt=(
            "你是一名资深产品经理。"
            "你负责把模糊需求整理成清晰 PRD。"
            "请输出：需求摘要、目标用户、核心功能、非功能需求、验收标准。"
            "内容要简洁、结构化、适合初学者阅读。"
            "优先使用项目符号，控制在 8 个要点以内。"
        ),
    )

    architect = create_agent(
        model=model,
        name="architect",
        system_prompt=(
            "你是一名软件架构师。"
            "你要根据需求设计整体方案。"
            "请输出：技术栈建议、系统结构、前后端职责、接口设计、潜在风险。"
            "尽量给出可落地的方案。"
            "优先使用项目符号，控制在 10 个要点以内。"
        ),
    )

    frontend_engineer = create_agent(
        model=model,
        name="frontend_engineer",
        system_prompt=(
            "你是一名前端工程师。"
            "你需要根据需求和架构输出前端实现方案。"
            "请输出：页面结构、组件拆分、状态管理、请求流程、一个简短代码骨架。"
            "使用 React 风格示例即可。"
            "除了代码骨架外，其余内容请尽量简短。"
            "代码骨架控制在 15 行以内。"
        ),
    )

    backend_engineer = create_agent(
        model=model,
        name="backend_engineer",
        system_prompt=(
            "你是一名后端工程师。"
            "你需要根据需求和架构输出后端实现方案。"
            "请输出：API 设计、数据流、错误处理、一个简短 FastAPI 风格代码骨架。"
            "除了代码骨架外，其余内容请尽量简短。"
            "代码骨架控制在 15 行以内。"
        ),
    )

    reviewer = create_agent(
        model=model,
        name="reviewer",
        system_prompt=(
            "你是一名严格但建设性的代码审查员。"
            "你会检查方案是否一致、是否存在风险、是否遗漏边界情况。"
            "请输出：通过点、问题点、修改建议。"
            "控制在 8 个要点以内。"
        ),
    )

    tester = create_agent(
        model=model,
        name="tester",
        system_prompt=(
            "你是一名测试工程师。"
            "你需要基于现有方案设计测试计划。"
            "请输出：关键测试场景、异常场景、接口测试点、UI/交互测试点。"
            "控制在 8 个要点以内。"
        ),
    )

    @tool
    def ask_product_manager(task: str) -> str:
        """让产品经理分析需求并输出 PRD。"""
        return run_subagent("产品经理 Agent", product_manager, task)

    @tool
    def ask_architect(task: str) -> str:
        """让架构师根据需求输出系统架构方案。"""
        return run_subagent("架构师 Agent", architect, task)

    @tool
    def ask_frontend_engineer(task: str) -> str:
        """让前端工程师输出前端实现方案和代码骨架。"""
        return run_subagent("前端工程师 Agent", frontend_engineer, task)

    @tool
    def ask_backend_engineer(task: str) -> str:
        """让后端工程师输出后端实现方案和代码骨架。"""
        return run_subagent("后端工程师 Agent", backend_engineer, task)

    @tool
    def ask_reviewer(task: str) -> str:
        """让审查员评估当前方案，指出风险和改进点。"""
        return run_subagent("代码审查 Agent", reviewer, task)

    @tool
    def ask_tester(task: str) -> str:
        """让测试工程师输出测试方案。"""
        return run_subagent("测试工程师 Agent", tester, task)

    supervisor = create_agent(
        model=model,
        name="engineering_supervisor",
        tools=[
            ask_product_manager,
            ask_architect,
            ask_frontend_engineer,
            ask_backend_engineer,
            ask_reviewer,
            ask_tester,
        ],
        system_prompt=(
            "你是一个软件团队的 supervisor agent。"
            "你的职责是协调多个专业子 agent 完成软件方案设计。"
            "对于每个需求，你必须按顺序调用以下工具："
            "1. ask_product_manager "
            "2. ask_architect "
            "3. ask_frontend_engineer "
            "4. ask_backend_engineer "
            "5. ask_reviewer "
            "6. ask_tester "
            "在调用后，综合所有结果输出最终总结。"
            "最终回答必须使用 Markdown，并包含以下标题："
            "## PRD"
            "## Architecture"
            "## Frontend"
            "## Backend"
            "## Review"
            "## Test Plan"
            "## Final Summary"
            "每个部分只做简短总结，不要重复粘贴全部原文。"
            "不要跳过任何一个工具。"
        ),
    )

    return supervisor


def run_demo(requirement: str) -> str:
    """执行完整的多 agent 协作流程。"""
    load_project_env()
    supervisor = build_team()

    print("=" * 60)
    print("LangChain 多智能体示例启动")
    print("=" * 60)
    print(f"需求: {requirement}")

    result = supervisor.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "请组织一个软件开发团队完成下面的需求分析与方案设计：\n"
                        f"{requirement}"
                    ),
                }
            ]
        }
    )

    final_answer = extract_text(result)

    print(f"\n{'=' * 16} Supervisor 最终总结 {'=' * 16}\n")
    print(final_answer)
    return final_answer


def main() -> None:
    requirement = "开发一个 AI 聊天机器人网站"
    if len(sys.argv) > 1:
        requirement = " ".join(sys.argv[1:])

    try:
        run_demo(requirement)
    except Exception as exc:
        print(f"\n运行失败: {exc}")
        print("请检查模型 API Key、网络连接，以及当前模型是否支持 tool calling。")
        raise


if __name__ == "__main__":
    main()
