"""
初始化 LangGraph 状态图用于 ETF 分析工作流。
"""

from langgraph.graph import StateGraph, END
from vflows.workflows.etf_analysis.state.schema import EtfAnalysisState
from vflows.workflows.etf_analysis.state.enums import Stage
from vflows.workflows.etf_analysis import nodes, routing


def create_etf_analysis_graph() -> StateGraph:
    """
    创建 ETF 分析工作流状态图。

    构建包含 5 个节点的线性工作流：
    init → fetch_csv → filter_a500 → analyze → output → END

    Returns:
        配置完成的 StateGraph 对象
    """
    workflow = StateGraph(EtfAnalysisState)

    # 添加节点
    workflow.add_node("init", nodes.init_node)
    workflow.add_node("fetch_csv", nodes.fetch_csv_node)
    workflow.add_node("filter_a500", nodes.filter_a500_node)
    workflow.add_node("analyze", nodes.analyze_node)
    workflow.add_node("output", nodes.output_node)

    # 设置入口
    workflow.set_entry_point("init")

    # 添加简单边（线性流程）
    workflow.add_edge("init", "fetch_csv")
    workflow.add_edge("fetch_csv", "filter_a500")

    # 条件边：检查过滤结果
    workflow.add_conditional_edges(
        "filter_a500",
        routing.check_filter_success,
        {
            "proceed": "analyze",
            "fail": "output",  # 未找到 ETF，直接输出错误
        }
    )

    workflow.add_edge("analyze", "output")
    workflow.add_edge("output", END)

    return workflow


# 全局缓存
_compiled_graph = None


def get_compiled_graph():
    """
    获取已编译的工作流图（单例模式）。

    使用全局缓存避免重复编译。

    Returns:
        已编译的图实例
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = create_etf_analysis_graph().compile()
    return _compiled_graph


async def run_etf_analysis(
    csv_url: str = "http://172.30.60.99:8688/ETFStats/etf_stats_top200.csv",
    target_index: str = "A500",
) -> EtfAnalysisState:
    """
    执行完整的 ETF 分析工作流。

    这是主要的入口函数，用于运行整个 ETF 分析流程。

    Args:
        csv_url: CSV 数据源 URL
        target_index: 目标指数名称（如 "A500"）

    Returns:
        包含所有输出的最终状态
    """
    from vflows.workflows.etf_analysis.state.schema import create_initial_state

    # 创建初始状态
    initial_state = create_initial_state(
        csv_url=csv_url,
        target_index=target_index,
    )

    # 获取已编译图并执行
    graph = get_compiled_graph()
    final_state = await graph.ainvoke(initial_state)

    return final_state
