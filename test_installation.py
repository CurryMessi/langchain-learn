import sys

# 检查Python版本
print(f"Python版本: {sys.version}")

# 尝试导入LangChain
try:
    import langchain
    print(f"✅ LangChain版本: {langchain.__version__}")
except ImportError as e:
    print(f"❌ LangChain导入失败: {e}")

# 尝试导入OpenAI集成
try:
    import langchain_openai
    print("✅ langchain-openai导入成功")
except ImportError as e:
    print(f"❌ langchain-openai导入失败: {e}")

# 尝试导入dotenv
try:
    import dotenv
    print("✅ python-dotenv导入成功")
except ImportError as e:
    print(f"❌ python-dotenv导入失败: {e}")
