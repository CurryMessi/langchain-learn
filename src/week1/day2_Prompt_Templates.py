"""
LangChain Day 2 实践：Prompt Templates高级用法

今天是实战演练日，我们将通过3个循序渐进的代码挑战任务，掌握LangChain中提示模板的核心技能：
1. 基础提示模板创建与使用
2. 聊天提示模板构建
3. 动态模板与部分填充

每个任务都包含详细注释和错误处理，适合"会一点Python"的学习者。
"""

import os
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# 加载环境变量
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
        ("langchain_core", "langchain-core"),
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
    
    print("✅ 环境检查通过！")
    return True


def task1_basic_prompt():
    """
    任务1：基础提示模板创建与使用
    
    目标：创建简单的PromptTemplate，体验变量替换的基本用法
    
    核心概念：
    - 变量插值：在模板字符串中使用{变量名}作为占位符
    - 模板实例化：使用PromptTemplate.from_template()创建模板对象
    - 格式化填充：使用format()方法将变量值填入模板
    
    学习重点：
    1. 理解模板如何分离格式与内容
    2. 掌握变量替换的基本语法
    3. 了解常见错误和排查方法
    """
    print("\n" + "="*50)
    print("🎯 任务1：基础提示模板")
    print("="*50)
    
    try:
        # 步骤1：定义模板字符串（包含变量占位符）
        # 这是一个解释技术概念的模板，有两个变量：
        # - concept: 要解释的概念名称
        # - level: 解释的难度级别
        template = "请用{level}难度的语言解释一下{concept}是什么"
        
        print(f"📝 模板内容: {template}")
        print("   🔹 变量: concept, level")
        
        # 步骤2：创建PromptTemplate实例
        # from_template()方法会自动从模板字符串中提取变量名
        prompt = PromptTemplate.from_template(template)
        
        # 查看提取到的变量名
        print(f"   🔹 提取的变量: {prompt.input_variables}")
        
        # 步骤3：格式化模板（填充变量）
        # format()方法接收关键字参数，将变量替换为具体值
        formatted_prompt = prompt.format(
            concept="大语言模型",
            level="初学者"
        )
        
        # 步骤4：显示结果
        print(f"\n✅ 格式化结果: {formatted_prompt}")
        
        # 步骤5：尝试不同值（扩展练习）
        print("\n🔧 扩展练习：尝试不同值")
        
        # 练习1：解释另一个概念
        prompt2 = prompt.format(concept="神经网络", level="中级")
        print(f"   练习1: {prompt2}")
        
        # 练习2：使用相同模板但不同难度
        prompt3 = prompt.format(concept="区块链", level="专家")
        print(f"   练习2: {prompt3}")
        
        print("\n🎉 任务1完成！你已掌握：")
        print("   1. ✅ 创建基础提示模板")
        print("   2. ✅ 使用变量插值")
        print("   3. ✅ 格式化模板生成完整提示")
        
        return formatted_prompt
        
    except Exception as e:
        print(f"❌ 任务1执行失败: {e}")
        print("\n💡 排查建议：")
        print("   1. 检查模板字符串中的花括号{}是否配对")
        print("   2. 确认format()方法传入的参数名与模板变量名一致")
        print("   3. 确保所有变量都提供了值")
        return None


def task2_chat_prompt():
    """
    任务2：聊天提示模板构建
    
    目标：使用ChatPromptTemplate创建带系统消息和用户消息的对话模板
    
    核心概念：
    - 多角色消息：system（系统指令）、human（用户输入）、ai（AI回复）
    - 消息列表：使用from_messages()方法构建对话流
    - 消息格式化：生成可直接传给聊天模型的消息对象列表
    
    学习重点：
    1. 理解聊天模型与普通语言模型的区别
    2. 掌握多角色对话模板的构建方法
    3. 了解消息对象的类型和用途
    """
    print("\n" + "="*50)
    print("🎯 任务2：聊天提示模板")
    print("="*50)
    
    try:
        # 步骤1：定义系统消息模板（角色设定）
        # 系统消息定义了AI的角色、能力和行为规则
        # 这里我们创建一个编程助手的角色
        system_template = "你是一个编程助手，擅长Python和算法，用简洁清晰的方式回答问题"
        
        # 步骤2：定义用户消息模板（问题输入）
        # 用户消息包含用户的提问，这里使用{question}作为变量
        human_template = "{question}"
        
        print("📋 消息模板定义：")
        print(f"   🔹 system: {system_template}")
        print(f"   🔹 human: {human_template}")
        print(f"   🔹 变量: question")
        
        # 步骤3：创建ChatPromptTemplate
        # from_messages()方法接收一个消息列表
        # 每个消息是一个元组：(角色, 模板内容)
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
        
        # 查看模板结构
        print(f"\n🔍 模板结构: {len(chat_prompt.messages)}条消息")
        for i, msg in enumerate(chat_prompt.messages):
            print(f"   消息{i+1}: {msg.prompt.input_variables}")
        
        # 步骤4：格式化模板
        # format_messages()返回消息对象列表，可直接传给聊天模型
        messages = chat_prompt.format_messages(
            question="如何优化冒泡排序？"
        )
        
        # 步骤5：打印格式化后的消息
        print("\n✅ 格式化消息列表：")
        for i, msg in enumerate(messages):
            print(f"   消息{i+1} [{msg.type}]: {msg.content}")
        
        # 步骤6：扩展练习 - 创建更复杂的对话模板
        print("\n🔧 扩展练习：创建多轮对话模板")
        
        # 创建包含AI回复历史的多轮对话模板
        multi_turn_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是算法专家，擅长解释算法原理和优化方法"),
            ("human", "第一轮问题：{question1}"),
            ("ai", "{answer1}"),
            ("human", "第二轮追问：{question2}")
        ])
        
        multi_turn_messages = multi_turn_prompt.format_messages(
            question1="什么是快速排序？",
            answer1="快速排序是一种分治算法，通过选取基准值将数组分为两部分...",
            question2="快速排序的时间复杂度是多少？"
        )
        
        print("   多轮对话示例（前2条消息）：")
        for i in range(2):
            print(f"     {multi_turn_messages[i].type}: {multi_turn_messages[i].content[:50]}...")
        
        print("\n🎉 任务2完成！你已掌握：")
        print("   1. ✅ 创建聊天提示模板")
        print("   2. ✅ 构建多角色对话场景")
        print("   3. ✅ 生成可直接传模型的消息列表")
        
        return messages
        
    except Exception as e:
        print(f"❌ 任务2执行失败: {e}")
        print("\n💡 排查建议：")
        print("   1. 检查角色名是否正确（只能使用system/human/ai/assistant）")
        print("   2. 确认模板字符串格式正确")
        print("   3. 确保所有变量都提供了值")
        return None


def task3_dynamic_template():
    """
    任务3：动态模板与部分填充
    
    目标：实现动态模板策略，使用partial()方法设置默认值
    
    核心概念：
    - 部分填充：使用partial()方法预先设置部分变量值
    - 动态策略：根据输入条件选择不同的模板或变量值
    - 模板复用：同一模板在不同场景下的灵活应用
    
    学习重点：
    1. 掌握partial()方法的用法和场景
    2. 理解模板策略模式的设计思路
    3. 学会根据需求动态生成提示
    """
    print("\n" + "="*50)
    print("🎯 任务3：动态模板策略")
    print("="*50)
    
    try:
        # 步骤1：创建基础模板（代码审查场景）
        # 这是一个通用的代码审查模板，有三个变量：
        # - language: 编程语言
        # - code: 要审查的代码
        # - focus: 审查的重点方面
        base_template = """请以{language}专家的身份审查以下代码：

代码：
{code}

审查重点：
{focus}

请指出：1. 潜在问题 2. 改进建议 3. 最佳实践应用"""
        
        print(f"📝 基础模板: {base_template[:100]}...")
        print("   🔹 变量: language, code, focus")
        
        # 步骤2：创建模板实例
        prompt = PromptTemplate.from_template(base_template)
        
        # 步骤3：创建部分填充模板（设置默认值）
        # partial()方法允许我们预先设置部分变量的值
        # 创建Python版本：设置默认语言和审查重点
        python_prompt = prompt.partial(
            language="Python",
            focus="代码规范(PEP8)、性能优化、异常处理、类型提示"
        )
        
        # 创建JavaScript版本：不同语言，不同审查重点
        javascript_prompt = prompt.partial(
            language="JavaScript",
            focus="异步处理、错误处理、ES6特性、浏览器兼容性"
        )
        
        print("\n🔧 部分填充模板创建完成：")
        print(f"   1. Python版本: language='Python', focus='代码规范...'")
        print(f"   2. JavaScript版本: language='JavaScript', focus='异步处理...'")
        
        # 步骤4：演示动态选择策略
        print("\n🎮 动态策略选择演示：")
        
        # 测试代码
        test_code = """def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total"""
        
        # 根据需求选择不同的模板策略
        print("   场景1：Python代码审查")
        python_result = python_prompt.format(code=test_code)
        print(f"     生成提示: {python_result[:150]}...")
        
        print("\n   场景2：JavaScript代码审查")
        js_code = """function fetchData(url) {
    return fetch(url)
        .then(response => response.json())
        .catch(error => console.error(error));
}"""
        js_result = javascript_prompt.format(code=js_code)
        print(f"     生成提示: {js_result[:150]}...")
        
        # 步骤5：演示更复杂的策略模式
        print("\n🔍 高级应用：模板工厂函数")
        
        def create_review_prompt(language="python", custom_focus=None):
            """根据语言和自定义重点创建审查提示"""
            
            # 内置策略库
            focus_strategies = {
                "python": "代码规范(PEP8)、性能优化、异常处理、类型提示",
                "javascript": "异步处理、错误处理、ES6特性、浏览器兼容性",
                "java": "面向对象设计、设计模式应用、内存管理、并发处理",
                "default": "代码结构、逻辑正确性、可读性、错误处理"
            }
            
            # 获取审查重点
            if custom_focus:
                focus = custom_focus
            else:
                focus = focus_strategies.get(language.lower(), focus_strategies["default"])
            
            # 创建并格式化提示
            return prompt.format(
                language=language.capitalize(),
                code=test_code,  # 使用测试代码
                focus=focus
            )
        
        # 测试工厂函数
        print("   测试模板工厂：")
        custom_prompt = create_review_prompt(language="python", custom_focus="算法复杂度分析")
        print(f"     自定义重点: {custom_prompt[:100]}...")
        
        print("\n🎉 任务3完成！你已掌握：")
        print("   1. ✅ 使用partial()方法设置默认值")
        print("   2. ✅ 实现动态模板策略")
        print("   3. ✅ 创建模板工厂函数")
        
        return python_result
        
    except Exception as e:
        print(f"❌ 任务3执行失败: {e}")
        print("\n💡 排查建议：")
        print("   1. 检查partial()方法传入的参数名是否与模板变量名一致")
        print("   2. 确保部分填充后，剩余变量在format()时提供值")
        print("   3. 注意可变对象的默认值问题")
        return None


def code_review_assistant():
    """
    综合实践：代码审查助手
    
    将今天学到的三个技能结合起来，构建一个简单的代码审查助手
    能够根据不同的编程语言和审查重点生成定制化的审查提示
    """
    print("\n" + "="*60)
    print("🚀 综合实践：代码审查助手")
    print("="*60)
    
    try:
        from langchain_core.prompts import PromptTemplate
        
        # 定义基础模板
        base_template = """请以{language}专家的身份审查以下代码：

代码：
{code}

审查重点：
{focus}

请按以下格式提供审查意见：
1. 潜在问题（按严重程度排序）
2. 具体改进建议
3. 可应用的最佳实践

注意：请用中文回答，保持专业且友好的语气。"""
        
        # 创建模板实例
        prompt = PromptTemplate.from_template(base_template)
        
        # 内置策略库：不同语言的审查重点
        focus_strategies = {
            "python": [
                "代码规范(PEP8)",
                "性能优化（时间/空间复杂度）",
                "异常处理（try-except）",
                "类型提示（type hints）",
                "函数设计（单一职责原则）"
            ],
            "javascript": [
                "异步处理（async/await, Promise）",
                "错误处理（try-catch, error boundaries）",
                "ES6+特性使用",
                "浏览器兼容性",
                "代码分割和懒加载"
            ],
            "java": [
                "面向对象设计原则",
                "设计模式应用",
                "内存管理和垃圾回收",
                "并发处理（多线程安全）",
                "异常处理策略"
            ]
        }
        
        def generate_review(language, code):
            """生成代码审查提示"""
            # 获取对应语言的审查重点
            if language.lower() in focus_strategies:
                focus_list = focus_strategies[language.lower()]
                focus = "、".join(focus_list)
            else:
                focus = "代码结构、逻辑正确性、可读性、错误处理"
            
            # 生成提示
            review_prompt = prompt.format(
                language=language.capitalize(),
                code=code,
                focus=focus
            )
            
            return review_prompt
        
        # 演示示例
        print("📊 代码审查助手演示")
        print("-"*40)
        
        # 示例1：Python代码审查
        python_code = """def find_duplicates(nums):
    duplicates = []
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] == nums[j]:
                duplicates.append(nums[i])
    return duplicates"""
        
        print("\n🔹 示例1：Python代码审查")
        print("   代码: 查找列表中的重复元素")
        python_review = generate_review("python", python_code)
        print(f"   生成提示长度: {len(python_review)}字符")
        print(f"   预览: {python_review[:120]}...")
        
        # 示例2：JavaScript代码审查
        js_code = """async function getUserData(userId) {
    try {
        const response = await fetch(`/api/users/${userId}`);
        return await response.json();
    } catch (error) {
        console.error('获取用户数据失败:', error);
        return null;
    }
}"""
        
        print("\n🔹 示例2：JavaScript代码审查")
        print("   代码: 异步获取用户数据")
        js_review = generate_review("javascript", js_code)
        print(f"   生成提示长度: {len(js_review)}字符")
        print(f"   预览: {js_review[:120]}...")
        
        # 示例3：自定义语言（使用默认策略）
        other_code = """func processData(data []int) int {
    sum := 0
    for _, value := range data {
        sum += value
    }
    return sum
}"""
        
        print("\n🔹 示例3：Go代码审查（使用默认策略）")
        print("   代码: 计算切片元素和")
        go_review = generate_review("go", other_code)
        print(f"   生成提示长度: {len(go_review)}字符")
        print(f"   预览: {go_review[:120]}...")
        
        print("\n🎯 实践总结：")
        print("   1. ✅ 成功创建了可复用的审查模板")
        print("   2. ✅ 实现了根据不同语言选择策略")
        print("   3. ✅ 构建了完整的代码审查助手框架")
        
        return {
            "python": python_review,
            "javascript": js_review,
            "go": go_review
        }
        
    except Exception as e:
        print(f"❌ 综合实践失败: {e}")
        return None


def main():
    """主函数：协调所有任务"""
    print("="*60)
    print("LangChain Day 2 实践：Prompt Templates高级用法")
    print("="*60)
    print("📚 今日学习目标：")
    print("   1. 掌握基础提示模板创建与使用")
    print("   2. 学会构建聊天提示模板")
    print("   3. 实现动态模板策略")
    print("   4. 完成综合实践：代码审查助手")
    print("-"*60)
    
    # 检查环境
    if not check_environment():
        print("\n⚠️ 环境未就绪，请先解决上述问题")
        return
    
    print("\n🚀 开始今日实践任务...")
    
    # 执行任务1
    task1_result = task1_basic_prompt()
    
    # 执行任务2
    task2_result = task2_chat_prompt()
    
    # 执行任务3
    task3_result = task3_dynamic_template()
    
    # 综合实践
    print("\n" + "="*60)
    print("🌟 挑战升级：综合实践")
    print("="*60)
    print("现在让我们将今天学到的所有技能结合起来，构建一个实用的代码审查助手！")
    
    review_results = code_review_assistant()
    
    # 总结
    print("\n" + "="*60)
    print("📋 今日成就总结")
    print("="*60)
    
    achievements = [
        ("✅", "基础提示模板", "掌握变量插值和格式化"),
        ("✅", "聊天提示模板", "构建多角色对话场景"),
        ("✅", "动态模板策略", "使用partial()和策略模式"),
        ("✅", "代码审查助手", "综合应用所有技能")
    ]
    
    for symbol, skill, detail in achievements:
        print(f"{symbol} {skill}: {detail}")
    
    print("\n🎯 核心技能掌握：")
    print("   1. 模板设计思维：分离格式与内容")
    print("   2. 变量管理：灵活处理动态输入")
    print("   3. 策略模式：根据需求选择不同模板")
    
    print("\n🔮 明日预告：")
    print("   Day 3将深入学习Memory管理，让AI记住对话历史！")
    
    print("\n💡 提醒：")
    print("   记得完成《今日学习卡片》，记录你的收获和疑问。")
    print("   如果有API密钥，可以尝试实际调用模型进行代码审查。")
    
    return {
        "task1": task1_result,
        "task2": task2_result,
        "task3": task3_result,
        "review": review_results
    }


if __name__ == "__main__":
    main()
