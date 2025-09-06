from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# 1. 定义工具 (Tools)
# 这里我们定义一个简单的工具，模拟天气查询
@tool
def get_weather(location: str):
    """根据地点获取天气信息"""
    # 模拟API调用
    return f"The weather in {location} is 25 degrees Celsius and sunny."

tools = [get_weather]

# 2. 定义LLM模型
# 我们使用 OpenAI 模型，并绑定之前定义的工具
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

# 3. 定义代理节点（Agent State）
# 这是一个处理所有逻辑的节点，它会调用LLM
def agent_node(state):
    # 'state'包含了所有上下文信息，比如用户输入和之前的工具输出
    # 我们可以用它来决定下一步做什么
    return llm.invoke(state['messages'])

# 4. 定义图
graph = StateGraph(str)  # 状态可以简单地是一个字符串（消息）
graph.add_node("agent", agent_node)
graph.add_node("tool", ToolNode(tools))

# 5. 定义路由
# 这是一个关键的函数，它根据LLM的输出决定下一步
def should_continue(state):
    # 检查LLM的输出中是否包含工具调用信息
    if state.tool_calls:
        return "continue_with_tool"
    else:
        return "end"

# 6. 设置图的边
graph.set_entry_point("agent")
graph.add_edge("tool", END)
graph.add_conditional_edges("agent", should_continue, {
    "continue_with_tool": "tool",
    "end": END
})

# 7. 编译并运行
runnable = graph.compile()
result = runnable.invoke("What is the weather in Paris?")

print(result)