"""
LangChain Day 1 实践：环境验证与基础使用
这是一个简单的Hello World示例，展示LangChain的基本用法
"""

import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# 1. 加载环境变量（从.env文件或系统环境变量）
if load_dotenv is not None:
    load_dotenv()

def check_environment():
    """检查运行环境是否就绪"""
    print("🔍 环境检查开始...")
    
    # 检查Python版本
    import sys
    print(f"   Python版本: {sys.version.split()[0]}")
    
    # 检查必要包
    required_packages = [
        ("langchain", "langchain"),
        ("langchain_openai", "langchain-openai"),
    ]
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"   ✅ {import_name} 已安装")
        except ImportError:
            print(f"   ❌ {import_name} 未安装，请运行: pip install {package_name}")
            return False

    if load_dotenv is None:
        print("   ⚠️ python-dotenv 未安装，将只读取系统环境变量")
        print("   如需从 .env 文件加载，请运行: pip install python-dotenv")
    
    # 检查API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("   ❌ DEEPSEEK_API_KEY 未设置")
        print("   请设置环境变量或创建.env文件")
        return False
    else:
        # 显示部分密钥（保护隐私）
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else " ***"
        print(f"   ✅ API密钥已设置: {masked_key}")
    
    print("✅ 环境检查通过！")
    return True

def hello_langchain():
    """使用LangChain进行简单的文本生成"""
    print("\n🤖 开始LangChain Hello World...")
    
    try:
        # 2. 导入必要的模块
        from langchain_openai import ChatOpenAI
        
        # 3. 初始化模型（使用gpt-3.5-turbo，成本较低）
        # temperature控制创造性：0=确定性高，1=创造性高
        llm = ChatOpenAI(
            model="deepseek-chat",          # DeepSeek模型名称
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),  # 显式传入API key
            temperature=0.7,
            max_tokens=100
        )
                
        # 4. 创建简单的提示
        prompt = "请用一句话介绍LangChain是什么，适合初学者理解"
        
        print(f"   问题: {prompt}")
        print("   等待模型响应...")
        
        # 5. 调用模型
        response = llm.invoke(prompt)
        
        # 6. 处理并显示结果
        answer = response.content.strip()
        print(f"\n   💡 模型回答: {answer}")
        
        return answer
        
    except Exception as e:
        print(f"   ❌ 发生错误: {e}")
        print("   可能的原因：")
        print("   1. API密钥无效或余额不足")
        print("   2. 网络连接问题")
        print("   3. 模型服务暂时不可用")
        return None

def simple_chain_example():
    """演示一个简单的链（Chain）用法"""
    print("\n⛓️ 尝试简单的链（Chain）示例...")
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # 1. 创建提示模板
        template = ChatPromptTemplate.from_messages([
            ("system", "你是一个友好的AI助手，专门帮助初学者学习技术概念"),
            ("user", "请用简单易懂的语言解释：{concept}")
        ])
        
        # 2. 初始化模型
        llm = ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0.7
        )
        
        # 3. 创建输出解析器
        output_parser = StrOutputParser()
        
        # 4. 构建链：template → llm → output_parser
        chain = template | llm | output_parser
        
        # 5. 运行链
        concepts = ["大语言模型", "提示工程", "机器学习"]
        for concept in concepts:
            print(f"   解释 '{concept}'...")
            result = chain.invoke({"concept": concept})
            print(f"   📘 {result[:80]}...")  # 只显示前80个字符
        
        print("✅ 链示例完成！")
        
    except Exception as e:
        print(f"   ❌ 链示例失败: {e}")
        print("   建议先确保基础示例能正常运行")

def main():
    """主函数：协调所有步骤"""
    print("=" * 50)
    print("LangChain Day 1 实践：概述与环境配置")
    print("=" * 50)
    
    # 步骤1：环境检查
    if not check_environment():
        print("\n⚠️ 环境未就绪，请先解决上述问题")
        print("   参考'环境配置指南'部分")
        return
    
    # 步骤2：Hello World示例
    result = hello_langchain()
    if result:
        print("\n🎉 恭喜！你已成功运行第一个LangChain应用")
        
        # 步骤3：简单链示例（可选）
        try:
            simple_chain_example()
        except:
            print("   链示例跳过，先专注于基础理解")
        
        # 总结
        print("\n" + "=" * 50)
        print("📋 今日收获总结")
        print("=" * 50)
        print("1. ✅ 验证了LangChain环境配置")
        print("2. ✅ 理解了基本组件：Models, Prompts")
        print("3. ✅ 体验了简单的文本生成流程")
        print("4. 🔄 下一步：深入学习Prompts模板和Chains")
        print("\n💡 提示：记得完成《今日学习卡片》记录你的收获！")
    else:
        print("\n⚠️ 示例运行失败，请检查错误信息并参考常见问题排查")

if __name__ == "__main__":
    main()
