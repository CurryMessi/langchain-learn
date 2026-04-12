"""
一个可运行的电商客服 Agent 示例。

适配当前项目安装的 LangChain 1.2.x：
- 使用 create_agent 创建真实 Agent
- 使用 @tool 注册工具
- 使用手动消息历史代替旧版 ConversationBufferMemory

运行方式：
    /Users/Curry/.pyenv/versions/3.10.13/bin/python src/agent/customer_service_agent.py
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage


ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT_DIR / ".env"

MOCK_ORDERS: Dict[str, Dict[str, object]] = {
    "12345": {
        "status": "已发货",
        "items": ["iPhone 15"],
        "total": 5999,
        "tracking_status": "快递运输中，预计明天下午送达",
    },
    "A20240401": {
        "status": "待出库",
        "items": ["机械键盘", "无线鼠标"],
        "total": 699,
        "tracking_status": "仓库正在打包，预计今天晚些时候发出",
    },
    "VIP888": {
        "status": "已签收",
        "items": ["AirPods Pro"],
        "total": 1899,
        "tracking_status": "已由前台代收，如有问题请联系人工客服",
    },
}

FAQ_DB = {
    "退货": "退货政策：签收后 7 天内支持无理由退货，商品需保持完好且配件齐全。",
    "退款": "退款说明：退货质检通过后，通常会在 1 到 3 个工作日原路退回。",
    "发货": "发货时间：工作日 16:00 前下单通常当天发出，节假日会顺延。",
    "支付": "支付方式：支持微信支付、支付宝、银行卡和信用卡。",
    "发票": "发票说明：下单时可申请电子发票，开票后会发送到预留邮箱。",
}

NEGATIVE_WORDS = ["生气", "不满意", "糟糕", "差", "垃圾", "投诉", "慢死了", "气死了"]


def load_project_env() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)


def build_model() -> ChatOpenAI:
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

    raise RuntimeError("未找到 API Key，请在 .env 中设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY。")


def extract_text(message) -> str:
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


def find_order_id(text: str) -> str | None:
    for order_id in MOCK_ORDERS:
        if order_id in text:
            return order_id

    match = re.search(r"[A-Za-z0-9]{5,}", text)
    if match:
        return match.group(0)

    return None


@tool
def query_order(order_text: str) -> str:
    """查询订单状态。输入订单号或包含订单号的文本，返回订单详情。"""
    order_id = find_order_id(order_text)
    if not order_id:
        return "没有识别到订单号，请让用户提供订单号后再查询。"

    order = MOCK_ORDERS.get(order_id)
    if not order:
        return f"未找到订单号 {order_id} 对应的订单，请核对后重试。"

    items = "、".join(order["items"])
    return (
        f"订单号：{order_id}；"
        f"状态：{order['status']}；"
        f"商品：{items}；"
        f"总金额：{order['total']}元；"
        f"物流信息：{order['tracking_status']}"
    )


@tool
def search_faq(question: str) -> str:
    """搜索常见问题知识库。输入问题文本，返回最相关的 FAQ 答案。"""
    for keyword, answer in FAQ_DB.items():
        if keyword in question:
            return answer
    return "未找到相关 FAQ，建议先向用户补充细节，必要时转人工客服。"


@tool
def detect_emotion(text: str) -> str:
    """检测用户情绪。输入用户原话，返回 negative 或 neutral。"""
    for word in NEGATIVE_WORDS:
        if word in text:
            return "negative"
    return "neutral"


class CustomerServiceBot:
    """一个带对话上下文的客服 Agent。"""

    def __init__(self) -> None:
        load_project_env()
        self.agent = create_agent(
            model=build_model(),
            tools=[query_order, search_faq, detect_emotion],
            system_prompt=(
                "你是一个友好、专业的电商客服 AI 助手。"
                "你的职责是回答用户问题、查询订单、解释常见政策，并在复杂场景下建议转人工。"
                "回答要求："
                "1. 始终礼貌、自然、简洁。"
                "2. 如果用户情绪可能不好，优先调用 detect_emotion。若结果是 negative，先安抚再处理问题。"
                "3. 涉及订单、物流、到货时间时，优先调用 query_order。"
                "4. 涉及退货、退款、发货、支付、发票等规则时，优先调用 search_faq。"
                "5. 只能根据工具返回的信息回答，不要编造订单状态。"
                "6. 如果缺少订单号，就礼貌地向用户索取。"
                "7. 无法确认或问题复杂时，建议转人工客服。"
            ),
            name="ecommerce_customer_service",
        )
        self.history: List[object] = []

    def chat(self, user_input: str, show_trace: bool = True) -> str:
        previous_len = len(self.history)
        result = self.agent.invoke(
            {
                "messages": self.history + [{"role": "user", "content": user_input}],
            }
        )
        self.history = result["messages"]

        if show_trace:
            self._print_trace(previous_len)

        return extract_text(self.history[-1])

    def _print_trace(self, previous_len: int) -> None:
        new_messages = self.history[previous_len:]
        for message in new_messages:
            if isinstance(message, ToolMessage):
                print(f"[工具结果] {extract_text(message)}")
            elif isinstance(message, AIMessage):
                tool_calls = getattr(message, "tool_calls", None) or []
                for call in tool_calls:
                    name = call.get("name", "unknown_tool")
                    args = call.get("args", {})
                    print(f"[调用工具] {name}({json.dumps(args, ensure_ascii=False)})")

    def clear_history(self) -> None:
        self.history = []


class EnhancedCustomerServiceAgent(CustomerServiceBot):
    """带情绪安抚和转人工兜底逻辑的增强版客服 Agent。"""

    HUMAN_TRANSFER_KEYWORDS = [
        "转人工",
        "人工客服",
        "不能解决",
        "复杂问题",
        "无法处理",
        "建议联系人工",
        "建议转人工",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.human_agent_queue: List[Dict[str, object]] = []

    def handle_message(self, user_input: str, show_trace: bool = True) -> str:
        comfort_prefix = self._build_comfort_prefix(user_input)

        try:
            result = self.chat(user_input, show_trace=show_trace)
            if self._need_human_agent(user_input, result):
                transfer_message = self._transfer_to_human(user_input, reason="rule_based_handoff")
                return f"{comfort_prefix}{transfer_message}" if comfort_prefix else transfer_message

            return f"{comfort_prefix}{result}" if comfort_prefix else result
        except Exception:
            transfer_message = self._transfer_to_human(user_input, reason="agent_error")
            return f"{comfort_prefix}{transfer_message}" if comfort_prefix else transfer_message

    def _build_comfort_prefix(self, user_input: str) -> str:
        emotion = detect_emotion.invoke({"text": user_input})
        if emotion == "negative":
            return "非常抱歉给您带来不好的体验，我先帮您处理这个问题。\n\n"
        return ""

    def _need_human_agent(self, user_input: str, agent_response: str) -> bool:
        combined_text = f"{user_input}\n{agent_response}"
        return any(keyword in combined_text for keyword in self.HUMAN_TRANSFER_KEYWORDS)

    def _transfer_to_human(self, user_input: str, reason: str) -> str:
        self.human_agent_queue.append(
            {
                "user_input": user_input,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "reason": reason,
            }
        )
        return "我先为您转接人工客服，请稍等片刻，我们会尽快帮您继续处理。"


def demo() -> None:
    bot = EnhancedCustomerServiceAgent()

    print("客服：您好！我是 AI 客服小助手，很高兴为您服务。")

    user1 = "我想查一下订单号12345的状态"
    print(f"用户：{user1}")
    print(f"客服：{bot.handle_message(user1)}\n")

    user2 = "什么时候能到？"
    print(f"用户：{user2}")
    print(f"客服：{bot.handle_message(user2)}\n")

    user3 = "如果我不满意，能退货吗？"
    print(f"用户：{user3}")
    print(f"客服：{bot.handle_message(user3)}\n")

    user4 = "你们这物流也太慢了，我有点生气"
    print(f"用户：{user4}")
    print(f"客服：{bot.handle_message(user4)}")


def interactive_chat() -> None:
    """命令行交互模式。"""
    bot = EnhancedCustomerServiceAgent()

    print("客服：您好！我是 AI 客服小助手，很高兴为您服务。")
    print("提示：输入 `exit` 或 `quit` 结束对话，输入 `clear` 清空上下文，输入 `queue` 查看待转人工队列。")

    while True:
        try:
            user_input = input("\n用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n客服：本次会话已结束，欢迎下次再来。")
            break

        if not user_input:
            print("客服：可以直接告诉我您的问题，比如“查一下订单号12345的状态”。")
            continue

        lowered = user_input.lower()
        if lowered in {"exit", "quit"}:
            print("客服：好的，本次服务先到这里，祝您生活愉快。")
            break

        if lowered == "clear":
            bot.clear_history()
            print("客服：上下文已经清空，我们可以重新开始。")
            continue

        if lowered == "queue":
            if not bot.human_agent_queue:
                print("客服：当前没有待转人工的会话。")
            else:
                print("客服：当前待转人工队列如下：")
                for item in bot.human_agent_queue:
                    print(
                        f"- 时间：{item['timestamp']} | 原因：{item['reason']} | 用户问题：{item['user_input']}"
                    )
            continue

        answer = bot.handle_message(user_input)
        print(f"客服：{answer}")


if __name__ == "__main__":
    interactive_chat()
