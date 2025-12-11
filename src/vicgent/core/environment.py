# %%
# GOTYOU
import os
from typing import Optional
from dotenv import load_dotenv
from vicgent.util.system import stopwatch
# Load environment variables from current directory
from dotenv import load_dotenv
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