from typing import Optional
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt.chat_agent_executor import AgentStatePydantic
from langgraph.graph import MessagesState

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

# --- The crucial part for LangServe input mapping ---
# Define a function to take the user's input and transform it into the initial AgentState
class InputDict(BaseModel):
    question:str