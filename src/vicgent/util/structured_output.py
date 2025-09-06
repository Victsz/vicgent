# GOTYOU
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import ToolMessage,HumanMessage
from typing import List, Type, Any, Dict
from pydantic import BaseModel

def make_structured_output(messages: List[BaseMessage], response_format: Type[BaseModel], llm_tool: Any) -> BaseModel:
    """
    使用LLM工具从消息历史中生成结构化输出。
    
    该函数通过绑定指定的Pydantic模型作为工具，引导LLM生成符合预定格式的结构化响应。
    仅使用最后一条消息作为上下文，通过工具调用机制确保输出严格遵循给定的数据模型。
    
    Args:
        messages: 消息历史列表，仅使用最后一条作为上下文
        response_format: 期望的输出格式，必须是Pydantic BaseModel子类
        llm_tool: 配置好的LLM工具实例
        
    Returns:
        BaseModel: 符合指定格式的结构化数据实例
        
    Raises:
        ValueError: 当模型未能正确调用工具或生成有效结构化数据时
    """
    messages = messages[-1:] # we only need the last message as context
    structured_response_schema = response_format
    llm_tool_ = llm_tool.bind_tools(tools=[structured_response_schema], tool_choice="any")
    hm = HumanMessage(content=f"Use the tool <{structured_response_schema.__name__}> to respond")
    messages.append(hm)
    re_try_count = 5
    for i in range(re_try_count):
        # 构建结构化响应
        try:
            response = llm_tool_.invoke(messages)
                # 提取结构化响应# 验证响应
            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                raise ValueError("模型没有调用任何工具")

            # 严格验证单工具调用
            if len(response.tool_calls) != 1:
                raise ValueError(f"期望单个工具调用，但收到 {len(response.tool_calls)} 个")

            tool_call = response.tool_calls[0]
            if tool_call["name"] != structured_response_schema.__name__:
                raise ValueError(f"期望调用 {structured_response_schema.__name__}，但调用了 {tool_call['name']}")

        
            structured_data = structured_response_schema(**tool_call["args"])
            return structured_data#{"structured_response": structured_data}
        except Exception as e:
            print(f"构建结构化响应失败 {i=}: {e}")
            
            # raise ValueError(f"构建结构化响应失败: {e}") from e
    raise ValueError(" 模型调用 structured output failed")

def gen_structured_output(messages: List[BaseMessage], response_format: Type[BaseModel], llm_tool: Any) -> Dict[str, Any]:

    structured_response_schema = response_format
    def find_tool_result(messages, tool_name):  
        """查找特定工具的执行结果"""  
        for message in reversed(messages):  # 从最新消息开始查找  
            if isinstance(message, ToolMessage)and (tool_name is None or message.name == tool_name):  
                return message  
        return None  
    tool_result = find_tool_result(messages, None)  
    assert tool_result is not None, "No tool result found"  
    final_message = messages[-1]
    llm_tool_ = llm_tool.bind_tools(tools=[structured_response_schema], tool_choice="any")
    hm = HumanMessage(content="Use the tool to respond to the user ")
    re_try_count = 5
    for i in range(re_try_count):
        response = llm_tool_.invoke([tool_result,final_message,hm])
            # 提取结构化响应# 验证响应
        if not hasattr(response, 'tool_calls') or not response.tool_calls:
            raise ValueError("模型没有调用任何工具")

        # 严格验证单工具调用
        if len(response.tool_calls) != 1:
            raise ValueError(f"期望单个工具调用，但收到 {len(response.tool_calls)} 个")

        tool_call = response.tool_calls[0]
        if tool_call["name"] != structured_response_schema.__name__:
            raise ValueError(f"期望调用 {structured_response_schema.__name__}，但调用了 {tool_call['name']}")

        # 构建结构化响应
        try:
            structured_data = structured_response_schema(**tool_call["args"])
            return {"structured_response": structured_data}
        except Exception as e:
            print(f"构建结构化响应失败 {i=}: {e}")
            # raise ValueError(f"构建结构化响应失败: {e}") from e
    raise ValueError(" 模型调用 structured output failed")
# %%
