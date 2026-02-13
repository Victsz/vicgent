"""
配置模块 - Claude API 和应用设置。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 获取项目根目录（向上查找直到找到 pyproject.toml 或 .git 目录）
current_dir = Path(__file__).resolve().parent
project_root = current_dir
while project_root.parent != project_root:
    if (project_root / "pyproject.toml").exists() or (project_root / ".git").exists():
        break
    project_root = project_root.parent

env_file = project_root / ".env"
if env_file.exists():    
    print(f"已加载环境配置: {env_file}")
else:
    # 回退到 .env.example
    env_example = project_root / ".env.example"
    if env_example.exists():
        env_file = env_example
        print(f"警告: 使用 .env.example (未找到 .env): {env_example}")

load_dotenv(env_file, verbose=True, override=True)
def get_model_id() -> str:
    """
    获取配置的 Claude 模型 ID。

    Returns:
        Claude 模型标识符（默认: claude-sonnet-4-20250514）
    """
    # model = os.getenv("ANTHROPIC_MODEL", "siliconflow,Pro/moonshotai/Kimi-K2-Thinking")
    model = os.getenv("ANTHROPIC_MODEL", "MiniMax,MiniMax-M2.1")
    print(f"模型标识符: {model}")
    return model


def get_api_key() -> str:
    """
    获取 Anthropic API 密钥。

    Returns:
        API 密钥字符串

    Raises:
        ValueError: 如果未设置 API 密钥
    """
    api_key = os.getenv("ANTHROPIC_API_KEY","")
    if not api_key:
        raise ValueError(
            "未找到 ANTHROPIC_API_KEY 环境变量。"
            "请设置 .env 文件或导出环境变量。"
        )
    print(f"API 密钥: {api_key}")
    return api_key


def get_agent_timeout_seconds() -> int:
    """
    获取智能体超时时间（秒）。

    从环境变量 AGENT_TIMEOUT_MINUTES 读取，默认为 90 分钟。

    Returns:
        超时时间（秒）
    """
    timeout_minutes = int(os.getenv("AGENT_TIMEOUT_MINUTES", "90"))
    return timeout_minutes * 60


def get_agent_batch_size() -> int:
    """
    获取智能体并行执行的批次大小。

    从环境变量 AGENT_BATCH_SIZE 读取，默认为 2。
    这控制研究阶段和分析阶段的并行智能体数量。

    Returns:
        批次大小（默认: 2）
    """
    batch_size = int(os.getenv("AGENT_BATCH_SIZE", "2"))
    # 限制在 1-5 之间（研究阶段最多5个agent）
    return max(1, min(batch_size, 5))


def get_output_base_dir() -> Path:
    """
    获取输出目录的基础路径。

    从环境变量 OUTPUT_BASE_DIR 读取，默认为 ./outputs

    Returns:
        输出目录的 Path 对象
    """
    output_dir = os.getenv("OUTPUT_BASE_DIR", "./outputs")
    return Path(output_dir)
