import os
import uuid
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

# 导入工具
from tools.weather import get_weather
from tools.calculator import calculator
from tools.translator import translate

# 加载环境变量
load_dotenv()

# 系统提示词
SYSTEM_PROMPT = """
你是小智，一个全能智能助手。

你可以帮用户：
• 查询天气：查询中国主要城市的天气
• 数学计算：进行各种数学运算
• 文本翻译：将中文翻译成其他语言

工作原则：
1. 先理解用户意图，选择合适的工具
2. 如果不确定，可以询问用户
3. 回答简洁明了，有帮助
"""

def create_assistant():
    """创建智能助手"""
    # 初始化模型
    llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
    max_tokens=100,
)
    
    # 创建记忆
    checkpointer = MemorySaver()
    
    # 创建 Agent
    agent = create_agent(
        model=llm,
        tools=[get_weather, calculator, translate],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer
    )
    
    return agent


def create_langfuse_tracing():
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        return None, None

    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
    )
    handler = CallbackHandler(public_key=public_key)
    return client, handler


def create_thread_id():
    return str(uuid.uuid4())

def main():
    """主函数"""
    print("=" * 50)
    print("      小智 - 全能智能助手 v1.0")
    print("=" * 50)
    print("输入 'quit' 退出\n")
    
    agent = create_assistant()
    langfuse_client, langfuse_handler = create_langfuse_tracing()
    if langfuse_handler is not None:
        print("Langfuse tracing: 已启用\n")
    else:
        print("Langfuse tracing: 未启用（缺少配置）\n")
    thread_id = create_thread_id()
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        user_input = input("你: ").strip()
        
        if user_input.lower() in ['quit', '退出', 'exit']:
            print("再见！")
            break
        
        if not user_input:
            continue
        
        try:
            invoke_config = dict(config)
            if langfuse_handler is not None:
                invoke_config["callbacks"] = [langfuse_handler]
                invoke_config["tags"] = ["smart-assistant", "cli", "deepseek"]
                invoke_config["run_name"] = "smart-assistant"
                invoke_config["metadata"] = {
                    "langfuse_session_id": thread_id,
                    "langfuse_user_id": "local-dev",
                }

            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                invoke_config
            )
            
            response = result["messages"][-1].content
            print(f"小智: {response}\n")

            if langfuse_client is not None:
                langfuse_client.flush()
            
        except Exception as e:
            print(f"出错了: {str(e)}\n")

    if langfuse_client is not None:
        langfuse_client.shutdown()

if __name__ == "__main__":
    main()
