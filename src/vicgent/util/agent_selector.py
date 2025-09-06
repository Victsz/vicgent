#!/usr/bin/env python3
"""
Agent Selector utility with factory pattern using langgraph's RunnableLambda.
Provides a factory function to create agent selector runnables.
"""

import os
from enum import Enum
from typing import Optional, Any, Callable, TypedDict, List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from vicgent.util.LLMUtil import create_model_anthropic
from vicgent.util.structured_output import make_structured_output

# Define the same SubAgent enum as in supervisor.py
class SubAgent(str, Enum):
    NOTSET = 'NOTSET'
    chat = 'chat'
    table_extractor_agent = 'table_extractor_agent'

class HandlerAgent(BaseModel):
    """Agent selection model - final output"""
    sub_agent: SubAgent = Field(
        description="The agent to handle task. Choose from available agents based on the task description.",
        default="NOTSET"
    )
    reason: Optional[str] = Field(
        description="The reason for selecting the agent. This can be used to provide additional context to the user.",
        default=None
    )

def create_agent_selector_factory(llm: Any) -> Callable:
    """
    Factory function to create an agent selector runnable.
    
    Args:
        llm: The LLM instance to use for agent selection
        
    Returns:
        Callable: A runnable function that takes task description and returns HandlerAgent
    """
    
    def agent_selector_runnable(task_description: str) -> HandlerAgent:
        """Runnable that selects appropriate agent for the given task"""
        
        # Prepare messages for structured output
        messages = [
            HumanMessage(content=f"""你是一个agent选择器。根据用户的任务描述，选择合适的子agent来处理。

用户任务: {task_description}

可用的子agents:
- table_extractor_agent: 专门处理图片表格提取任务
- chat: 通用聊天agent
- NOTSET: 默认选项，当没有明确匹配的agent时使用

请选择合适的agent来处理这个任务, 回答格式:
选择:[table_extractor_agent|chat|NOTSET]
理由: [10字以内的原因描述]
""")
        ]
        
        try:
            # 1. Need to create a general response 
            rsp = llm.invoke(messages[-1].content)
            print(f"Response: {rsp.content}")
            messages.append(rsp)
            
            # 2. Generate structured output using the utility function
            result = make_structured_output(
                messages=messages,
                response_format=HandlerAgent,
                llm_tool=llm
            )
            return result
            
        except Exception as e:
            print(f"Error selecting agent: {e}")
            # Return default agent on error
            return HandlerAgent(sub_agent=SubAgent.NOTSET)
    
    # Wrap with RunnableLambda for langgraph compatibility
    return RunnableLambda(agent_selector_runnable)

def create_agent_selector_graph_factory(llm: Any) -> StateGraph:
    """
    Factory function to create a complete agent selector graph.
    
    Args:
        llm: The LLM instance to use for agent selection
        
    Returns:
        StateGraph: A compiled StateGraph for agent selection
    """
    
    # Create the agent selector runnable
    agent_selector = create_agent_selector_factory(llm)
    
    # Define state type for the graph
    
    class AgentSelectionState(TypedDict):
        task_description: str
        selected_agent: Optional[HandlerAgent]
        messages: List[BaseMessage]
    
    # Define the graph nodes
    def agent_node(state: AgentSelectionState):
        """Agent node that processes the task and selects agent"""
        selected_agent = agent_selector.invoke(state["task_description"])
        return {
            "selected_agent": selected_agent,
            "messages": state.get("messages", []) + [HumanMessage(content=state["task_description"])]
        }
    
    # Build the graph
    graph = StateGraph(AgentSelectionState)
    graph.add_node("agent_selector", agent_node)
    graph.set_entry_point("agent_selector")
    graph.add_edge("agent_selector", END)
    
    return graph.compile()

# Example usage function
def get_agent_selector(llm: Any = None) -> Callable:
    """
    Get an agent selector runnable. If no LLM is provided, creates one using default config.
    
    Args:
        llm: Optional LLM instance. If None, creates one using LLMUtil
        
    Returns:
        Callable: Agent selector runnable function
    """
    if llm is None:
        load_dotenv("/home/victor/workspace/playgrounds/langchain/.agent.env", override=True)
        llm = create_model_anthropic(
            model_name=os.getenv("TMODEL", "Pro/deepseek-ai/DeepSeek-V3"),
            temperature=0
        )
    
    return create_agent_selector_factory(llm)

if __name__ == "__main__":
    """Simple test for the agent selector factory"""
    selector = get_agent_selector()
    result = selector.invoke("用户上传了一个包含表格的图片，需要提取表格内容")
    print(f"Selected agent: {result.sub_agent}, Reason: {result.reason}")