# %%
# GOTYOU
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import MessagesState
from langgraph.prebuilt.chat_agent_executor import AgentStatePydantic
# %%
import os
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_core.globals import set_debug
from langchain_core.output_parsers import StrOutputParser
set_debug(True)
load_dotenv("/home/victor/workspace/my_steerings/study_langchain/src/study_langchain/playground.env")
parser = StrOutputParser()
MODEL = os.getenv("LLM_MODEL")

MODEL = "zai-org/GLM-4.5V"
MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
vllm = init_chat_model(
    model=MODEL,
    api_key=os.getenv("API_KEY"),
    # base_url=os.getenv("API_BASE"),
    temperature=0, # critical for tool calling structureoutput
    model_provider="anthropic",  # Changed from "anthropic" to "openai"


)
# print("Testing model connection...")
# # response = vllm.invoke("are you ready? answer with yes or no")
# chain = (vllm|parser)
# rsp_content = chain.invoke("are you ready? answer with yes or no")

# print(f"Model response: {rsp_content=}")
# %%
tllm = init_chat_model(
    # model="Pro/deepseek-ai/DeepSeek-R1",
    model = "Pro/deepseek-ai/DeepSeek-V3",
    # model="Pro/moonshotai/Kimi-K2-Instruct",
    # model = "zai-org/GLM-4.5V",
    api_key=os.getenv("API_KEY"),
    # base_url=os.getenv("API_BASE"),
    model_provider="anthropic",  # Changed from "anthropic" to "openai"
    temperature = 0,
)
# print("Testing model connection...")
# # response = tllm.invoke("are you ready? answer with yes or no")
# chain = (tllm|parser)
# rsp_content = chain.invoke("are you ready? answer with yes or no")

# print(f"Model response: {rsp_content=}")
# %%
from typing import Optional

from langchain_core.messages import HumanMessage  , AIMessage
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
from util.file_util import load_image,save_markdown_table
from util.structured_output import gen_structured_output,gen_structured_output2
# file_path = "/home/victor/workspace/my_steerings/test_img.png"
# in_messages = [f"帮我提取{file_path}的表格"]

def call_model(state: AgentState):
    # breakpoint()
    messages = [
        # 让LLM知道使用什么工具很重要
        {"role": "system", "content": "你可以负责提取文件的路径, 不需要进一步处理"},

    ]

    to_get_final = state.get("original_file",None) is not None

    in_messages = state["messages"]
    user_request = in_messages[-1]
    # input={"messages": [("human", "what's the weather in sf?")]}


    messages.append(HumanMessage(content=user_request.content))
    # breakpoint()
    file_rsp:FileResponse = gen_structured_output2(messages=messages,response_format= FileResponse,llm_tool=tllm)
    # Return only the state keys you want to update
    if to_get_final:
        return {"final_response":file_rsp}
    return {"original_file": file_rsp}


def extract_table(state:AgentState):

    image = state["original_file"]

    image_msg = HumanMessage(content= load_image(image.path))
    extract_messages = [
        # 让LLM知道使用什么工具很重要
        {"role": "system", "content": "你负责完整提取表格, 直接输出markdown源码即可"},
    ]
    extract_messages.append(image_msg)
    # return_msg:AIMessage = vllm.invoke(extract_messages)
    chain = (vllm|parser)
    table_str = chain.invoke(extract_messages)
    breakpoint()
    return {"table_str":table_str}

def store_table(state:AgentState):
    table_str = state["table_str"]

    save_messages = [
        # 让LLM知道使用什么工具很重要
        {"role": "system", "content": "你负责使用工具save_markdown_table保存markdown表格文件, 只需要保存markdown源码, 并基于内容补充标题.同时以标题作为文件名"},
    ]
    tbl_msg = HumanMessage(content=f"请保存下面的表格\n\n{table_str}")
    save_messages.append(tbl_msg)


    tools = [save_markdown_table]

    # 创建工具字典
    tools_dict = {tool.name: tool for tool in tools}
    for i in range(5):
        try:
            save_rsp:AIMessage = tllm.bind_tools(tools=tools,tool_choice="any").invoke(save_messages)
            def validate():
                if not save_rsp.tool_calls:
                    return False
                if len(save_rsp.tool_calls) != 1:
                    return False
                tc = save_rsp.tool_calls[0]
                selected_tool = tc["name"]
                if selected_tool != "save_markdown_table":
                    return False
                return True
            if not validate():
                raise ValueError("")
            # 2. 执行工具
            tc = save_rsp.tool_calls[0]

            selected_tool = tools_dict[tc["name"]]
            tool_msg = selected_tool.invoke(tc)
            save_messages.append(save_rsp)
            save_messages.append(tool_msg)

            # final_messages = [
            # # 让LLM知道使用什么工具很重要
            #     {"role": "system", "content": "总结工具使用结果, 不需要进一步询问."},
            #     save_rsp,
            #     tool_msg,
            #     HumanMessage(content="文件完整保存路径?")
            # ]
            # chain = (tllm|parser)
            # final_msg = chain.invoke(final_messages)
            return {"messages":[HumanMessage(content=f"帮我提取路径: <{tool_msg.content}>")]}
            break
        except:
            pass
    raise Exception(f"store table failed")
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

from langchain_core.runnables import RunnableLambda,RunnablePassthrough
def create_final_reponse(state:AgentState):

    fr:FileResponse = state.get("final_response", None)
    msg =  f"Failed to extract image"
    table_str = state.get("table_str", None)
    if fr is not None:
        msg= f"Table saved to {fr.path} content\n\n{table_str}"
    elif table_str is not None:
        msg= f"Failed to save content\n\n{table_str}"
    return {
        "messages":[AIMessage(content=msg)]
    }


graph = workflow.compile() | RunnableLambda(create_final_reponse)
print(graph.get_graph().draw_ascii()) # uv add grandalf
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
        "messages":[HumanMessage(content=input_dict.get("question", ""))]
    }

# Use RunnableLambda to create a runnable that initializes the state.
# initial_state_runnable = RunnablePassthrough.assign(
#     initial_state=create_initial_state
# )
# Use RunnableLambda to directly map the input to the state
initial_state_runnable = RunnableLambda(create_initial_state)


full_graph_with_input =  (
    initial_state_runnable |
    graph |
    RunnableLambda(create_final_reponse)
    ).with_types(input_type=InputDict, output_type=str)

# Now, chain your initial_state_runnable with your actual graph.
# The graph will receive the fully-formed AgentState.

if __name__ == '__main__':
    file_path = "/home/victor/workspace/my_steerings/test_img.png"
    in_messages = [f"帮我提取{file_path}的表格"]

    g_res = graph.invoke(input={"messages":in_messages})
    print(g_res)