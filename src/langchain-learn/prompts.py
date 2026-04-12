from datetime import datetime
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# 定义模板，用 {} 占位
template = PromptTemplate.from_template(
    "请用{language}语言介绍一下{topic}，不超过100字。"
)

# 定义多轮对话模板
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一位{role}，擅长用简洁易懂的方式解释复杂概念。"),
    ("human", "请解释一下：{concept}"),
])

# 生成消息列表
messages = chat_template.format_messages(
    role="物理学教授",
    concept="量子纠缠"
)


# 填充变量
prompt = template.format(language="中文", topic="人工智能")
print(prompt)

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("未找到 DEEPSEEK_API_KEY，请在项目根目录的 .env 文件或系统环境变量中设置。")

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=api_key,
    temperature=0.7,
    max_tokens=100,
)

# 调用模型
response = llm.invoke(messages)
print(response.content)


# 定义示例 few_shot_template，用于few-shot learning
examples = [
    {"input": "开心", "output": "我今天非常开心！"},
    {"input": "难过", "output": "我感到有些难过..."},
]


# 创建包含示例的模板
few_shot_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个情绪表达助手。"),
    ("human", "示例：\n情绪: 开心\n表达: 我今天非常开心！\n\n情绪: 难过\n表达: 我感到有些难过..."),
    ("human", "现在请表达这个情绪: {emotion}")
])

messages = few_shot_template.format_messages(emotion="兴奋")
response = llm.invoke(messages)

# 动态日期模板
template = PromptTemplate(
    template="今天是{date}，请告诉我关于{topic}的最新消息。",
    input_variables=["topic"],
    partial_variables={
        "date": datetime.now().strftime("%Y年%m月%d日")  # 自动填充
    }
)

# 只需要传 topic
prompt = template.format(topic="AI发展")
