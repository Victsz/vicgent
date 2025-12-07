# %%
# GOTYOU
import os
from typing import Optional
from dotenv import load_dotenv
from vicgent.util.system import stopwatch
# Load environment variables from current directory
env_file = "/home/victor/workspace_local/agent_services/.agent.env"
load_dotenv(env_file,override=False,verbose=True)

antropic_base_url = os.getenv("ANTHROPIC_BASE_URL")
api_key = os.getenv("API_KEY")
import logging

log_method = lambda x: logging.info(x) if len(logging.getLogger().handlers) > 0 else print(x)

os.environ['ANTHROPIC_AUTH_TOKEN'] = api_key
log_method(f"ANTHROPIC_BASE_URL: {antropic_base_url}")

# breakpoint()  # Commented out for demo
# breakpoint()
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_debug
from langchain_core.runnables import RunnableLambda

from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.prebuilt.chat_agent_executor import AgentStatePydantic

from vicgent.util.file_util import load_image, save_markdown_table
from vicgent.util.structured_output import make_structured_output
# Environment setup
set_debug(True)
parser = StrOutputParser()

# Debug: Check if environment variables are loaded correctly
log_method(f"ANTHROPIC_BASE_URL: {os.getenv('ANTHROPIC_BASE_URL')}")
log_method(f"API_KEY present: {bool(os.getenv('API_KEY'))}")
log_method(f"VMODEL: {os.getenv('VMODEL')}")
log_method(f"TMODEL: {os.getenv('TMODEL')}")
# Visual model for image processing
VMODEL = os.getenv("VMODEL", "Qwen/Qwen2.5-VL-72B-Instruct")
# Text model for tool calling
TMODEL = os.getenv("TMODEL", "Pro/deepseek-ai/DeepSeek-V3")
# ANTHROPIC_BASE_URL="https://api.siliconflow.cn" # SF_URL
# os.environ['ANTHROPIC_BASE_URL'] = ANTHROPIC_BASE_URL
# Initialize visual model for image extraction
vllm = init_chat_model(
    model=VMODEL,
    # load from os.environ['ANTHROPIC_AUTH_TOKEN'] 
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
    temperature=0,  # critical for tool calling structure output
    model_provider="anthropic",
)
# %%
# Initialize text model for tool calling
tllm = init_chat_model(
    model=TMODEL,
    # load from os.environ['ANTHROPIC_AUTH_TOKEN'] 
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
    model_provider="anthropic",
    temperature=0,
)
log_method("Testing model connection...")
# response = tllm.invoke("are you ready? answer with yes or no")
chain = (tllm|parser)
# 使用方式
with stopwatch("模型调用"):
    rsp_content = chain.invoke("are you ready? answer with yes or no")
log_method(f"Model response: {rsp_content=}") 
# breakpoint()
# %%

# Inherit 'messages' key from MessagesState, which is a list of chat messages
class FileResponse(BaseModel):
    name:str
    path:str
    description:str

class AgentState_Safe(AgentStatePydantic):
    # Final structured response from the agent
    original_file:Optional[FileResponse] = None
    final_response: Optional[FileResponse] = None
    table_str:Optional[str] = None

class AgentState(MessagesState):
    # Final structured response from the agent
    original_file:Optional[FileResponse] = None
    final_response: Optional[FileResponse] = None
    table_str:Optional[str] = None
# %%

# Define the function that calls the model
# file_path = "/home/victor/workspace/my_steerings/test_img.png"
# in_messages = [f"帮我提取{file_path}的表格"]

def call_model(state: AgentState):
    # breakpoint()
    messages = [
        # 让LLM知道使用什么工具很重要
        {"role": "system", "content": "你可以负责提取文件的路径, 不需要进一步处理"},

    ]

    to_get_final = state.get("original_file",None) is not None

    messages_list = state["messages"]
    user_request = messages_list[-1]
    # input={"messages": [("human", "what's the weather in sf?")]}


    messages.append(HumanMessage(content=user_request.content))
    # breakpoint()
    with stopwatch("FormatedInput"):
        file_rsp:FileResponse = make_structured_output(messages=messages,response_format= FileResponse,llm_tool=tllm)
    # Return only the state keys you want to update
    if to_get_final:
        return {"final_response":file_rsp}
    return {"original_file": file_rsp}
def validate_tool_call(response):
    if not response.tool_calls:
        error_msg = "没有工具调用"
        tool_call_id = "error_call_id"
        error_message = ToolMessage(
            content=f"工具调用失败: {error_msg}",
            tool_call_id=tool_call_id,
            status="error"
        )
        return False, error_message
    
    if len(response.tool_calls) != 1:
        error_msg = f"期望单个工具调用，但收到 {len(response.tool_calls)} 个"
        tool_call_id = response.tool_calls[0]["id"] if response.tool_calls else "error_call_id"
        error_message = ToolMessage(
            content=f"工具调用失败: {error_msg}",
            tool_call_id=tool_call_id,
            status="error"
        )
        return False, error_message
    
    tc = response.tool_calls[0]
    selected_tool = tc["name"]
    if selected_tool != "save_markdown_table":
        error_msg = f"期望调用 save_markdown_table，但调用了 {selected_tool}"
        error_message = ToolMessage(
            content=f"工具调用失败: {error_msg}",
            tool_call_id=tc["id"],
            status="error"
        )
        return False, error_message
    
    return True, None

def extract_table(state:AgentState):

    image = state["original_file"]

    image_msg = HumanMessage(content= load_image(image.path))
    extract_messages = [
        # 让LLM知道使用什么工具很重要
        {"role": "system", "content": "你负责完整提取表格直输出markdown源码即可"},
    ]
    extract_messages.append(image_msg)
    # return_msg:AIMessage = vllm.invoke(extract_messages)
    chain = (vllm|parser)
    with stopwatch("extract_table"):
        table_str = chain.invoke(extract_messages)
        # breakpoint()
        if table_str.strip() == "" or len(table_str.strip().split("\n")) < 2: # dirty fix for empty table
            extract_messages = [
            # 让LLM知道使用什么工具很重要
                {"role": "system", "content": "你负责提取图片里的文字信息"},
            ]
            image_msg = HumanMessage(content= load_image(image.path,table=False))
            extract_messages.append(image_msg)
            log_method(f"No table detected, trying to extract text instead")
            table_str = chain.invoke(extract_messages)
    
    # breakpoint()  # Removed debug statement from production code
    return {"table_str":table_str}

def store_table(state:AgentState):
    table_str = state["table_str"]
    state_safe = AgentState_Safe.model_validate(state)
    org_name = state_safe.original_file.name
    save_messages = [
        # 让LLM知道使用什么工具很重要
        {"role": "system", "content": f"你负责使用工具save_markdown_table保存markdown<表格>文件. 如果<没有表格>, 你需要提取关键信息通常为['时间','指标名称','比率', '绝对数', '说明'],<整理>出一份<表格>. 然后保存<表格>的<markdown源码>, 并基于**{org_name=}**补充标题. 特殊情况: 如果org_name意义不明, 结合表格内容生成标题. 同时以标题作为<文件名>"},
    ]
    tbl_msg = HumanMessage(content=f"请保存下面的表格\n<表格>\n{table_str}\n<\表格>")
    save_messages.append(tbl_msg)


    tools = [save_markdown_table]

    # 创建工具字典
    tools_dict = {tool.name: tool for tool in tools}
    for _ in range(5):
        try:
            save_rsp:AIMessage = tllm.bind_tools(tools=tools,tool_choice="any").invoke(save_messages)
             
            re, error_tool_msg = validate_tool_call(save_rsp)
            if not re:
                
                # 将错误消息添加到消息历史中
                save_messages.append(save_rsp)
                save_messages.append(error_tool_msg) # let model know the error
                
                raise ValueError(f"Tool call validation failed: {error_tool_msg.content}")
            
            # Execute tool
            tc = save_rsp.tool_calls[0]
            selected_tool = tools_dict[tc["name"]]
            tool_msg = selected_tool.invoke(tc)
            save_messages.append(save_rsp)
            save_messages.append(tool_msg)

            return {"messages":[HumanMessage(content=f"帮我提取路径: <{tool_msg.content}>")]}
            
        except Exception as e:
            import traceback
            log_method(f"Tool execution attempt failed: {e}\n{traceback.format_exc()}")
            continue
    
    raise RuntimeError("store table failed after 5 attempts")
# %%
# Define the function that determines whether to continue or not
def should_continue(state: AgentState):
    # breakpoint()
    if state.get("final_response", None) is not None:
        return "end"
    if state.get("original_file", None) is not None:
        return "begin"


workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between

workflow.add_node("agent", call_model)
workflow.add_node("extract_table", extract_table)
workflow.add_node("store_table", store_table) # try tool node

# We now add a conditional edge
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "begin": "extract_table",
        "end": END,
    },
)

workflow.add_edge("extract_table", "store_table")
workflow.add_edge("store_table", "agent")

def create_final_reponse(state:AgentState):
    
    fr:FileResponse = state.get("final_response", None)
    msg = "Failed to extract image"
    table_str = state.get("table_str", None)
    # breakpoint()
    if fr is not None:
        msg = f"Table saved to {fr.path} content\n\n{table_str}"
    elif table_str is not None:
        msg = f"Failed to save content\n\n{table_str}"
    return {
        "messages":[AIMessage(content=msg)]
    }

def create_final_reponse_safe(state:AgentState):
    
    state_safe = AgentState_Safe.model_validate(state)
    return state_safe

graph = workflow.compile() | RunnableLambda(create_final_reponse)
log_method(graph.get_graph().draw_ascii()) # uv add grandalf - requires grandalf package
# %%

# %%
# --- The crucial part for LangServe input mapping ---
# Define a function to take the user's input and transform it into the initial AgentState
class InputDict(BaseModel):
    question:str
def create_initial_state(input_dict: InputDict) -> AgentState:
    """Transforms a simple input dict into the initial AgentState."""
    # breakpoint()
    return {
        "messages":[HumanMessage(content=input_dict.question)]
    }

# Use RunnableLambda to create a runnable that initializes the state.
# initial_state_runnable = RunnablePassthrough.assign(
#     initial_state=create_initial_state
# )
# Use RunnableLambda to directly map the input to the state
initial_state_runnable = RunnableLambda(create_initial_state)


full_graph_with_input =  (
    initial_state_runnable |
    graph 
    # |
    # RunnableLambda(create_final_reponse)
    ).with_types(input_type=InputDict, output_type=AgentState)

# Now, chain your initial_state_runnable with your actual graph.
# The graph will receive the fully-formed AgentState.

if __name__ == '__main__':
    file_path = "/home/victor/workspace_local/agent_services/AgentDocsSpace/extract_table/input/processed/error.1761446598314_d..png"
    in_message = f"帮我提取{file_path}的表格"

    g_res = full_graph_with_input.invoke(input={"question":in_message})
    log_method(g_res)