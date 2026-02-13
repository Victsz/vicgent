"""
类型安全的枚举类，用于避免硬编码字符串。
"""

from enum import Enum


class StateField(str, Enum):
    """
    EtfAnalysisState 中的所有字段名。

    用于类型安全地访问状态字典，避免拼写错误。
    """
    # 输入
    CSV_URL = "csv_url"
    TARGET_INDEX = "target_index"

    # 数据
    RAW_CSV_DATA = "raw_csv_data"
    FILTERED_ETFS = "filtered_etfs"

    # 分析结果
    ANALYSIS_RESULTS = "analysis_results"

    # 输出
    OUTPUT_JSON = "output_json"

    # 元数据
    CURRENT_STAGE = "current_stage"
    ERRORS = "errors"


class Stage(str, Enum):
    """
    工作流执行阶段。

    用于跟踪工作流执行状态。
    """
    INIT = "init"
    FETCH_COMPLETE = "fetch_complete"
    FILTER_COMPLETE = "filter_complete"
    ANALYSIS_COMPLETE = "analysis_complete"
    OUTPUT_COMPLETE = "output_complete"
    COMPLETE = "complete"
    FAILED = "failed"
