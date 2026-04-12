# LCEL链式组合示例
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的AI助手，使用中文回答"),
    ("human", "{query}")
])

# 创建模型实例
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat"
)

# 组合成链
chain = prompt | llm | StrOutputParser()

# 调用链
result = chain.invoke({"query": "请用一句话介绍LangChain"})
print(f"AI回复: {result}")