#!/usr/bin/env python3
"""
演示 extractor.py 中两种图版本的返回差异
1. 带 RunnableLambda(create_final_reponse) 的版本
2. 不带 RunnableLambda(create_final_reponse) 的版本
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
# project_root = Path(__file__).parent.parent.parent.parent
# sys.path.insert(0, str(project_root))

from vicgent.core.extractor import (
    workflow, AgentState, InputDict, create_initial_state, create_final_reponse,create_final_reponse_safe,
    RunnableLambda
)
from vicgent.util.system import stopwatch

def demo_with_final_response():
    """演示带 create_final_reponse 的图版本"""
    print("=" * 60)
    print("演示 1: 带 RunnableLambda(create_final_reponse) 的版本")
    print("=" * 60)
    
    # 创建带 create_final_reponse 的图
    graph_with_final = workflow.compile() | RunnableLambda(create_final_reponse)
    
    # 创建初始状态
    input_dict = InputDict(question="帮我提取 /home/victor/workspace_local/agent_services/AgentDocsSpace/extract_table/input/processed/2021F_PPMT_RevenueIP.png 的表格")
    initial_state = create_initial_state(input_dict)
    
    print("输入:", input_dict.question)
    print("\n开始处理...")
    
    try:
        with stopwatch("带 create_final_reponse 的图执行"):
            result = graph_with_final.invoke(initial_state)
        
        print("\n返回结果类型:", type(result))
        print("返回结果结构:")
        for key, value in result.items():
            if key == "messages":
                print(f"  {key}: {type(value)} (长度: {len(value)})")
                for i, msg in enumerate(value):
                    print(f"    消息 {i}: {type(msg).__name__}")
                    print(f"    内容: {msg.content[:100]}..." if len(msg.content) > 100 else f"    内容: {msg.content}")
            else:
                print(f"  {key}: {type(value)} = {value}")
                
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()

def demo_with_final_response_safe():
    """演示带 create_final_reponse_safe 的图版本"""
    print("\n" + "=" * 60)
    print("演示 2: 带 RunnableLambda(create_final_reponse_safe) 的版本")
    print("=" * 60)
    
    # 创建带 create_final_reponse_safe 的图
    graph_without_final = workflow.compile()| RunnableLambda(create_final_reponse_safe)
    
    # 创建初始状态
    input_dict = InputDict(question="帮我提取 /home/victor/workspace_local/agent_services/AgentDocsSpace/extract_table/input/processed/2021F_PPMT_RevenueIP.png 的表格")
    initial_state = create_initial_state(input_dict)
    
    print("输入:", input_dict.question)
    print("\n开始处理...")
    
    try:
        with stopwatch("带 create_final_reponse_safe 的图执行"):
            result = graph_without_final.invoke(initial_state)
         
        print("\n返回结果类型:", type(result))
        print("返回结果结构:")
         
                
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()

def demo_comparison():
    """对比两种版本的差异"""
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    
    print("""
1. 带 RunnableLambda(create_final_reponse) 的版本:
   - 返回简化的消息格式
   - 适合直接向用户展示结果
   - 文件路径信息嵌入在消息文本中
   - 返回结构: {"messages": [AIMessage(content="用户友好的消息")]}

2. 带 RunnableLambda(create_final_reponse_safe) 的版本:
   - 返回完整的状态对象
   - 适合程序进一步处理
   - 提供结构化的数据访问
   - 返回结构: {
       "messages": [...完整消息历史...],
       "original_file": FileResponse(...),
       "final_response": FileResponse(...),
       "table_str": "提取的表格内容"
     }

3. 使用建议:
   - 如果需要直接向用户展示结果，使用带 create_final_reponse 的版本
   - 如果需要程序化处理结果，使用带 create_final_reponse_safe 的版本
   - 可以根据具体需求选择合适的版本
    """)

if __name__ == "__main__":
    print("VicGent Extractor 演示程序")
    print("展示两种图版本的返回差异")
    
    # 演示两种版本
    # demo_with_final_response()
    demo_with_final_response_safe()
    demo_comparison()