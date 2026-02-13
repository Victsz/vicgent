"""
工作流节点函数 - ETF 分析工作流的各个步骤。

所有节点都是异步函数，接收状态并返回更新。
"""

from typing import Dict, Any
from vflows.workflows.etf_analysis.state.schema import EtfAnalysisState
from vflows.workflows.etf_analysis.state.enums import StateField, Stage


async def init_node(state: EtfAnalysisState) -> Dict[str, Any]:
    """
    初始化节点。

    准备执行环境，记录开始状态。

    Args:
        state: 当前工作流状态

    Returns:
        包含 current_stage 更新的字典
    """
    print("=" * 60)
    print("[ETF Analysis Workflow] 启动")
    print("=" * 60)
    print(f"CSV URL: {state[StateField.CSV_URL]}")
    print(f"目标指数: {state[StateField.TARGET_INDEX]}")
    print()

    return {
        StateField.CURRENT_STAGE: Stage.FETCH_COMPLETE
    }


async def fetch_csv_node(state: EtfAnalysisState) -> Dict[str, Any]:
    """
    获取 CSV 数据节点（桩实现）。

    Phase 1: 模拟数据获取，打印日志并返回模拟数据。

    Args:
        state: 当前工作流状态

    Returns:
        包含 raw_csv_data 和 current_stage 更新的字典
    """
    print("Running: fetch_csv_node")

    csv_url = state[StateField.CSV_URL]

    # Phase 1 桩实现：仅记录，不实际获取
    print(f"  [STUB] 将从 URL 获取 CSV: {csv_url}")
    print(f"  [STUB] 模拟获取了 200 行数据")

    # 模拟 CSV 数据（前几行用于展示）
    mock_csv = """ETF代码,基金名称,跟踪指数,规模,日期
159300,A500ETF,中证A500,100.5,2024-01
159301,A500基金,中证A500,50.2,2024-01
510300,沪深300ETF,沪深300,200.3,2024-01
"""

    return {
        StateField.RAW_CSV_DATA: mock_csv,
        StateField.CURRENT_STAGE: Stage.FETCH_COMPLETE,
    }


async def filter_a500_node(state: EtfAnalysisState) -> Dict[str, Any]:
    """
    过滤 A500 ETF 节点（桩实现）。

    Phase 1: 模拟过滤逻辑，返回模拟的 ETF 列表。

    Args:
        state: 当前工作流状态

    Returns:
        包含 filtered_etfs 和 current_stage 更新的字典
    """
    print("Running: filter_a500_node")

    target_index = state[StateField.TARGET_INDEX]
    csv_data = state.get(StateField.RAW_CSV_DATA, "")

    # Phase 1 桩实现：直接返回模拟数据
    print(f"  [STUB] 过滤跟踪指数: {target_index}")
    print(f"  [STUB] 输入 CSV 数据长度: {len(csv_data)} 字符")

    mock_filtered_etfs = [
        {
            "ETF代码": "159300",
            "基金名称": "A500ETF",
            "跟踪指数": "中证A500",
            "规模": "100.5",
            "日期": "2024-01"
        },
        {
            "ETF代码": "159301",
            "基金名称": "A500基金",
            "跟踪指数": "中证A500",
            "规模": "50.2",
            "日期": "2024-01"
        }
    ]

    print(f"  [STUB] 找到 {len(mock_filtered_etfs)} 个匹配的 ETF")

    return {
        StateField.FILTERED_ETFS: mock_filtered_etfs,
        StateField.CURRENT_STAGE: Stage.FILTER_COMPLETE,
    }


async def analyze_node(state: EtfAnalysisState) -> Dict[str, Any]:
    """
    分析 A500 ETF 节点（桩实现）。

    Phase 1: 模拟分析逻辑，为每个 ETF 生成模拟分析结果。

    Args:
        state: 当前工作流状态

    Returns:
        包含 analysis_results 和 current_stage 更新的字典
    """
    print("Running: analyze_node")

    filtered_etfs = state.get(StateField.FILTERED_ETFS, [])

    # Phase 1 桩实现：生成模拟分析结果
    print(f"  [STUB] 分析 {len(filtered_etfs)} 个 ETF")

    analysis_results = []
    for etf in filtered_etfs:
        mock_analysis = {
            "etf_code": etf["ETF代码"],
            "etf_name": etf["基金名称"],
            "recommendation": "买入" if etf["ETF代码"] == "159300" else "持有",
            "confidence": 0.85,
            "reasoning": f"[STUB] 基于 {etf['基金名称']} 的规模和流动性分析"
        }
        analysis_results.append(mock_analysis)
        print(f"    - {etf['基金名称']}: {mock_analysis['recommendation']}")

    return {
        StateField.ANALYSIS_RESULTS: analysis_results,
        StateField.CURRENT_STAGE: Stage.ANALYSIS_COMPLETE,
    }


async def output_node(state: EtfAnalysisState) -> Dict[str, Any]:
    """
    输出结果节点。

    将分析结果格式化为 JSON 并输出。

    Args:
        state: 当前工作流状态

    Returns:
        包含 output_json 和 current_stage 更新的字典
    """
    print("Running: output_node")

    import json

    analysis_results = state.get(StateField.ANALYSIS_RESULTS, [])

    # 构建最终输出
    output = {
        "target_index": state[StateField.TARGET_INDEX],
        "total_etfs_found": len(state.get(StateField.FILTERED_ETFS, [])),
        "analysis_results": analysis_results,
        "status": "success"
    }

    output_json = json.dumps(output, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print("最终输出 (JSON)")
    print("=" * 60)
    print(output_json)
    print("=" * 60)
    print()

    return {
        StateField.OUTPUT_JSON: output_json,
        StateField.CURRENT_STAGE: Stage.COMPLETE,
    }
