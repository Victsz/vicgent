"""
ETF 分析工作流的状态模式。
"""

from typing import TypedDict, List, Optional, Annotated
from operator import add
from .enums import Stage, StateField


class EtfAnalysisState(TypedDict):
    """
    贯穿 ETF 分析工作流的核心状态对象。
    """

    # 输入
    csv_url: str
    target_index: str  # e.g., "A500"

    # 数据（顺序更新，无需 reducer）
    raw_csv_data: Optional[str]
    filtered_etfs: Optional[List[dict]]  # ETFs matching target index

    # 分析结果（如果是并行分析可能需要 reducer，但 Phase 1 顺序执行）
    analysis_results: Optional[List[dict]]

    # 输出
    output_json: Optional[str]

    # 元数据
    current_stage: str
    errors: Annotated[List[str], add]  # 多个节点可能添加错误


def create_initial_state(
    csv_url: str = "http://172.30.60.99:8688/ETFStats/etf_stats_top200.csv",
    target_index: str = "A500",
) -> EtfAnalysisState:
    """
    创建初始工作流状态。

    Args:
        csv_url: CSV 数据源 URL
        target_index: 目标指数名称（如 "A500"）

    Returns:
        初始化的 EtfAnalysisState 字典
    """
    return {
        StateField.CSV_URL: csv_url,
        StateField.TARGET_INDEX: target_index,
        StateField.RAW_CSV_DATA: None,
        StateField.FILTERED_ETFS: None,
        StateField.ANALYSIS_RESULTS: None,
        StateField.OUTPUT_JSON: None,
        StateField.CURRENT_STAGE: Stage.INIT,
        StateField.ERRORS: [],
    }
