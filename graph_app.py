# %%

import os

from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langgraph.prebuilt.chat_agent_executor import create_react_agent, AgentState
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    HumanMessage
)
from pydantic import BaseModel
BASE_URL = "https://api.deepseek.com/v1"
DEEP_SK_API_KEY=os.getenv('ANTHROPIC_AUTH_TOKEN')
MODEL = os.getenv('ANTHROPIC_MODEL')
class WeatherResponse(BaseModel):
    conditions: str
    city: str

checkpointer = InMemorySaver()
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"{city}-晴"
# %%
# init model
os.environ['ANTHROPIC_AUTH_TOKEN'] = DEEP_SK_API_KEY
os.getenv('ANTHROPIC_MODEL')
llm = init_chat_model(
    model='deepseek-reasoner',
    base_url=os.getenv('ANTHROPIC_BASE_URL').strip(),
    # key is retrived from env variables
    # api_key=os.getenv('ANTHROPIC_AUTH_TOKEN'),
    model_provider="anthropic",
    temperature=0,
)
# response = llm.invoke("Are you ready? Yes or No?")
# print(response)
# %%

def prompt_g(state: AgentState):
    in_messages = state["messages"]
    last_message = in_messages[-1]
    if isinstance(last_message, ToolMessage):
        import traceback
        traceback.print_stack()
        # breakpoint()
        return in_messages
    messages = [
        SystemMessage(content="你可以通过get_weather函数获取天气信息.直接给我答案"),
        # 如果需要，可以在这里添加 LLM 调用结果
        # res = llm.invoke([message])  # 您提到的需求
    ]
    #messages.extend()
    # user_input = state.get("input", None)
    # if user_input:
    #     print(user_input)
    #     messages.append(HumanMessage(content=user_input))
    # else:
    messages.append(HumanMessage(content=state["messages"][-1].content))
    # messages.append(HumanMessage(content="你可以通过get_weather函数获取天气信息.直接给我答案"),)
    return messages

agent = create_react_agent(
    model=llm, # anthropic works while openai failed
    tools=[get_weather],
    prompt=prompt_g,
    # checkpointer=checkpointer,
    response_format=WeatherResponse,
    debug=True,
    
)
app = agent


# config = {"configurable": {"thread_id": "1"}}
# res = agent.invoke({"messages": "what is the weather in Shenzhen"},config=config)
# print(res)
# Run the agent
# agent.invoke(
#     {"messages": [{"role": "user", "content": "what is the weather in Shenzhen"}]},
#     config=config
# )
# %%



# --- The crucial part for LangServe input mapping ---
# Define a function to take the user's input and transform it into the initial AgentState
class InputDict(BaseModel):
    question: str

def create_initial_state(input_dict: InputDict) -> AgentState:
    """Transforms a simple input dict into the initial AgentState."""
    return AgentState(
        messages=[HumanMessage(content=input_dict.get("question", ""))],        
    )
 
from langchain_core.runnables import RunnableLambda
# Use RunnableLambda to directly map the input to the state
initial_state_runnable = RunnableLambda(create_initial_state)

# Now, chain your initial_state_runnable with your actual graph.
# The graph will receive the fully-formed AgentState.
def get_response(state: AgentState):    
    # return "good"
    # breakpoint()
    return state["structured_response"]#["structured_response"]

full_graph_with_input = initial_state_runnable | agent | RunnableLambda(get_response)
# Wrap your composition with explicit schemas  
full_graph_with_input = (  
    initial_state_runnable   
    | agent   
    | RunnableLambda(get_response)
).with_types(input_type=InputDict, output_type=WeatherResponse)
# full_graph_with_input.invoke({"question": "what is the weather in Shenzhen"})