"""
工作流路由函数 - 条件分支决策逻辑。

Phase 1: 简单线性工作流，暂无复杂路由需求。
此文件为未来扩展预留。
"""

from typing import Literal
from vflows.workflows.etf_analysis.state.schema import EtfAnalysisState
from vflows.workflows.etf_analysis.state.enums import StateField


def check_filter_success(
    state: EtfAnalysisState,
) -> Literal["proceed", "fail"]:
    """
    检查过滤是否成功。

    Phase 1: 简单检查是否有找到 ETF。

    Args:
        state: 当前工作流状态

    Returns:
        - "proceed": 找到 ETF，继续分析
        - "fail": 未找到 ETF，终止流程
    """
    filtered_etfs = state.get(StateField.FILTERED_ETFS)

    if filtered_etfs and len(filtered_etfs) > 0:
        return "proceed"

    return "fail"
