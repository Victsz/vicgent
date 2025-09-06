# GOTYOU
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import ToolMessage,HumanMessage
def gen_structured_output2(messages:list[BaseMessage],response_format,llm_tool):
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

def gen_structured_output(messages:list[BaseMessage],response_format,llm_tool):
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
