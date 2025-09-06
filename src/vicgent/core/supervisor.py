# %%
#GOTYOU
"""
Supervisor Agent - 管理多个专门化的agent
基于 Reference/supervisor_agent.md 实现
遵循 Linus 哲学：简洁、实用、消除特殊情况
使用官方推荐的节点包装方案解决 ParentCommand 错误
"""

from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from enum import Enum
load_dotenv("/home/victor/workspace/playgrounds/langchain/.agent.env",override=True)
from vicgent.util.LLMUtil import create_model_anthropic
# Load environment


# 定义任务完成判断的结构化输出
class CompareResult(BaseModel):
    """任务完成判断结果"""
    completed: bool = Field(description="任务是否已完成，True表示完成，False表示未完成")
    why_no_message: str = Field(description="如果未完成，说明原因和需要改进的地方", default="")

# Import agent selector factory instead of defining locally
from ..util.agent_selector import SubAgent, HandlerAgent, create_agent_selector_factory

# 扩展的Supervisor状态，包含agent选择信息
class SupervisorState(MessagesState):
    """Supervisor状态，继承MessagesState并添加handler_agent字段"""
    handler_agent: HandlerAgent = Field(default_factory=lambda: HandlerAgent())

def create_supervisor():
    """创建supervisor系统 - 使用官方推荐的节点包装方案"""

    # 导入table extractor agent和状态类型
    from vicgent.core.extractor import graph as table_extractor_agent, AgentState_Safe

    # 初始化LLM
    llm = create_model_anthropic(model_name=os.getenv("TMODEL"),
                                 temperature=0)

    # 创建agent选择器
    agent_selector = create_agent_selector_factory(llm)
    
    def agent_selection_node(state: SupervisorState):
        """Agent选择节点 - 使用factory模式自动选择agent"""
        breakpoint()
        if not state["messages"]:
            return {"handler_agent": HandlerAgent(sub_agent=SubAgent.NOTSET)}
        
        # 提取最后一个用户消息作为任务描述
        last_message = state["messages"][-1]
        if hasattr(last_message, 'content'):
            task_description = str(last_message.content)
            
            # 使用factory选择agent
            selected_agent = agent_selector.invoke(task_description)
            breakpoint()
            return {"handler_agent": selected_agent}
        
        return {"handler_agent": HandlerAgent(sub_agent=SubAgent.NOTSET)}

    # 不再需要单独的supervisor agent，直接使用agent_selector进行决策
    # agent_selector已经包含了智能选择逻辑

    def extract_file_path_from_messages(messages):
        """从消息中提取文件路径"""
        for msg in reversed(messages):  # 从最新消息开始查找
            if hasattr(msg, 'content'):
                content = str(msg.content)
                # 简单的文件路径提取逻辑
                if '.png' in content or '.jpg' in content or '.jpeg' in content:
                    # 提取路径
                    import re
                    path_match = re.search(r'[^\s]+\.(png|jpg|jpeg)', content)
                    if path_match:
                        return path_match.group(0)
        return None

    def should_route_to_agent(state: SupervisorState):
        """基于handler_agent选择路由到合适的agent"""
        breakpoint()
        handler_agent = state.get("handler_agent")
        
        if handler_agent and handler_agent.sub_agent == SubAgent.table_extractor_agent:
            return "table_extractor"
        elif handler_agent and handler_agent.sub_agent == SubAgent.chat:
            return "chat_agent"
        else:
            return END  # 没有匹配的agent或NOTSET，结束对话

    # 🎯 关键：包装函数直接调用子图
    def call_table_extractor(state: SupervisorState):
        """包装函数：直接调用table_extractor子图"""
        breakpoint()
        # 提取文件路径
        file_path = extract_file_path_from_messages(state["messages"])
        message_content = f"帮我提取{file_path}的表格" if file_path else "请处理表格提取任务"
        new_user_message = HumanMessage(content=message_content)

        # 创建AgentState_Safe (Pydantic模型)
        new_state = AgentState_Safe(
            messages=[new_user_message],
        )

        # 转换Pydantic模型为typed dict - 简单！
        table_extractor_state = new_state.model_dump()

        # 直接调用子图
        result = table_extractor_agent.invoke(table_extractor_state)

        # 返回转换后的结果，保持handler_agent状态
        return {
            "messages": result.get("messages", []),
            "handler_agent": state.get("handler_agent")
        }

    def call_chat_agent(state: SupervisorState):
        """聊天agent节点 - 简单回应"""
        last_message = state["messages"][-1] if state["messages"] else None
        
        if last_message and hasattr(last_message, 'content'):
            # 简单回应
            from langchain_core.messages import AIMessage
            response = AIMessage(content=f"收到您的聊天请求: {last_message.content}")
            return {
                "messages": state["messages"] + [response],
                "handler_agent": state.get("handler_agent")
            }
        
        return state

    # 智能任务完成判断函数 - 保留原始核心逻辑！
    def should_continue(state: SupervisorState):
        """使用LLM智能判断任务是否完成"""
        messages = state["messages"]

        if not messages:
            return "supervisor"

        # 获取原始用户请求（第一个human消息）
        original_request = None
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == "human":
                original_request = msg.content
                break

        # 获取最后的消息（sub_agent的返回）
        last_message = messages[-1]
        if not hasattr(last_message, 'content') or not original_request:
            return "supervisor"

        # 使用LLM进行智能判断
        try:
            from ..util.structured_output import gen_structured_output2

            judge_messages = [
                HumanMessage(content=f"""你是负责比对任务完成结果的专家。

原始用户需求：{original_request}

Sub-agent的最后返回：{last_message.content}

请判断sub-agent的返回是否完成了用户的原始需求。如果没有完成，请说明原因和需要改进的地方。""")
            ]

            result: CompareResult = gen_structured_output2(
                messages=judge_messages,
                response_format=CompareResult,
                llm_tool=llm
            )

            if result.completed:
                return END
            else:
                # 如果未完成，可以记录原因用于重新分配
                print(f"任务未完成，原因: {result.why_no_message}")
                return "supervisor"

        except Exception as e:
            print(f"智能判断失败，使用简单规则: {e}")
            # 降级到简单字符串匹配
            content = str(last_message.content)
            if "Saved markdown table to" in content or "保存" in content:
                return END
            return "supervisor"

    # 构建多agent图 - 简化架构，直接使用agent_selector进行决策
    supervisor_graph = (
        StateGraph(SupervisorState)  # 使用 SupervisorState
        .add_node("agent_selector", agent_selection_node)  # Agent选择节点（包含决策逻辑）
        .add_node("table_extractor", call_table_extractor) # 表格提取agent
        .add_node("chat_agent", call_chat_agent)           # 聊天agent
        .add_edge(START, "agent_selector")                 # 从选择器开始
        .add_conditional_edges(
            "agent_selector",
            should_route_to_agent,  # 基于选择的agent直接路由
            {
                "table_extractor": "table_extractor",
                "chat_agent": "chat_agent",
                END: END
            }
        )
        .add_conditional_edges(
            "table_extractor",
            should_continue,
            {
                "agent_selector": "agent_selector",  # 返回选择器重新评估
                END: END,
            }
        )
        .add_edge("chat_agent", END)  # 聊天完成后直接结束
        .compile()
    )

    return supervisor_graph

# 主要接口
def get_supervisor_agent():
    """获取编译好的supervisor agent"""
    return create_supervisor()

print(get_supervisor_agent().get_graph().draw_ascii())  # uv add grandalf
# %%
if __name__ == "__main__":
    supervisor = get_supervisor_agent()
    print(supervisor.get_graph().draw_ascii())  # uv add grandalf

    # 测试supervisor
    test_input = SupervisorState(
        messages=[
            HumanMessage(content="帮我提取/home/victor/workspace/playgrounds/langchain/test_data/blood.png的表格")
        ],
        handler_agent=HandlerAgent()
    )

    print("Testing supervisor agent...")
    try:
        result = supervisor.invoke(test_input)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")