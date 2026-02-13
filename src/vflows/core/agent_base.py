"""
结构化输出智能体基础模块 - 使用 Pydantic 模型返回类型安全的结构化数据。
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Type, TypeVar

# uv add/ pip install - [claude-agent-sdk,pydantic]
from pydantic import BaseModel
from claude_agent_sdk import ( 
    query,
    ClaudeAgentOptions,
    ResultMessage,
    AssistantMessage,
    UserMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
    ThinkingBlock,
)

# 配置日志
logger = logging.getLogger(__name__)


def print_env():
    """打印关键环境变量配置"""
    env_vars = [
        # Anthropic API 配置
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "NO_PROXY",
        "DISABLE_TELEMETRY",
        "DISABLE_COST_WARNINGS",
        "API_TIMEOUT_MS",
        # # Agent 配置
        # "AGENT_TIMEOUT_MINUTES",
        # "AGENT_BATCH_SIZE",
        # "OUTPUT_BASE_DIR",
        # # LangSmith 配置
        # "LANGCHAIN_TRACING_V2",
        # "LANGCHAIN_API_KEY",
        # "LANGCHAIN_PROJECT",
    ]

    logger.info("=" * 60)
    logger.info("环境变量配置:")
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # 对敏感信息进行脱敏
            if "KEY" in var or "TOKEN" in var:
                display_value = value[:8] + "..." if len(value) > 8 else "***"
            else:
                display_value = value
            logger.info(f"  {var}: {display_value}")
        else:
            logger.debug(f"  {var}: (未设置)")
    logger.info("=" * 60)


# ============================================================================
# 核心类定义
# ============================================================================

class AgentResult(BaseModel):
    """来自任意智能体调用的标准化结果"""

    success: bool
    output: Optional[Any] = None
    raw_output: Optional[str] = None
    error: Optional[str] = None
    agent_name: str
    execution_time_ms: int


# ============================================================================
# 工具函数
# ============================================================================


def set_output_directory(task_name: str, _output_base_dir=Path("./outputs")):
    """设置输出目录（工作流开始时调用一次）"""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 基于环境变量配置的基础目录创建任务目录
    _output_base_dir = _output_base_dir / f"{task_name}_{timestamp}"
    _output_base_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📁 输出目录: {_output_base_dir}")
    return _output_base_dir

def _save_agent_output(result: AgentResult, output_folder: Path):
    """
    保存单个 agent 的输出到文件。

    Args:
        result: Agent 执行结果
        output_folder: 输出目录路径
    """
    if output_folder is None:
        return

    try:
        # 确保输出目录存在
        output_folder.mkdir(parents=True, exist_ok=True)

        agent_name = result.agent_name

        # 保存解析后的 JSON
        if result.success and result.output:
            json_path = output_folder / f"{agent_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                # 如果是 Pydantic 模型，使用 model_dump()
                if isinstance(result.output, BaseModel):
                    json.dump(result.output.model_dump(), f, indent=2, ensure_ascii=False)
                else:
                    json.dump(result.output, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 已保存: {output_folder.name}/{agent_name}.json")

        # 保存原始输出
        if result.raw_output:
            raw_path = output_folder / f"{agent_name}_raw.txt"
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(result.raw_output)
    except Exception as e:
        logger.warning(f"⚠️  保存失败 {result.agent_name}: {e}")


T = TypeVar('T', bound=BaseModel)
async def run_agent_structured(
    agent_name: str,
    prompt: str,
    model:str,
    response_model: Type[T],
    tools: Optional[List[str]] = None, 
    system_prompt: Optional[str] = None,
    timeout_seconds: int = 60,
    phase: str = "research",
    debug: bool = False,
    output_base_dir: Path = Path("./outputs")
) -> AgentResult:
    """
    运行 Claude Agent SDK 调用并返回结构化输出（Pydantic 模型）。
    """
    start_time = time.time()
    try:
        output_folder = set_output_directory(f"{agent_name}-{phase}", _output_base_dir=output_base_dir)
        # 1. 准备请求配置
        schema = response_model.model_json_schema()
        options = ClaudeAgentOptions(
            model=model,
            allowed_tools=tools,
            permission_mode="acceptEdits",
            cwd=output_folder,
            output_format={"type": "json_schema", "schema": schema},  # 关键：设置为JSON Schema格式
            
            sandbox={
                "enabled": True,
                "excludedCommands": [],  # 仅 git 需要访问宿主机
                "allowUnsandboxedCommands": False,  # 其他命令必须在沙箱
            },
        )

        # 2. 打印环境变量并执行查询
        print_env()
        messages = query(prompt=prompt, options=options)

        # 3. 处理消息流，获取结构化输出（带超时控制）
        result_message = None

        async def process_messages():
            nonlocal result_message
            async for message in messages:
                # Debug 模式：输出所有类型的消息
                if debug:
                    if isinstance(message, AssistantMessage):
                        logger.debug(f"[{agent_name}] 🤖 AssistantMessage:")
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                logger.debug(f"  📝 Text: {block.text[:200]}...")
                            elif isinstance(block, ToolUseBlock):
                                logger.debug(f"  🔧 Tool: {block.name} (id: {block.id})")
                                if block.input:
                                    logger.debug(f"     Input: {str(block.input)[:100]}...")
                            elif isinstance(block, ThinkingBlock):
                                logger.debug(f"  💭 Thinking: {block.thinking[:100]}...")

                    elif isinstance(message, UserMessage):
                        logger.debug(f"[{agent_name}] 👤 UserMessage (echo)")

                    elif isinstance(message, SystemMessage):
                        logger.debug(f"[{agent_name}] ⚙️  SystemMessage: {message.subtype}")

                # 捕获最终的 ResultMessage
                if isinstance(message, ResultMessage):
                    result_message = message
                    if debug:
                        logger.debug(f"[{agent_name}] ✅ ResultMessage received")
                        logger.debug(f"  📊 Turns: {message.num_turns}, Duration: {message.duration_ms}ms")
                        if message.total_cost_usd:
                            logger.debug(f"  💰 Cost: ${message.total_cost_usd:.4f}")

        # 使用超时控制执行
        try:
            await asyncio.wait_for(process_messages(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return AgentResult(
                success=False,
                error=f"Agent execution timed out after {timeout_seconds} seconds",
                agent_name=agent_name,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )

        # 4. 验证结果
        if result_message is None:
            return AgentResult(
                success=False,
                error="No ResultMessage received from agent",
                agent_name=agent_name,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )

        if result_message.is_error:
            return AgentResult(
                success=False,
                error=result_message.result or "Agent returned an error",
                raw_output=result_message.result,
                agent_name=agent_name,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )

        # 5. 解析结构化输出为 Pydantic 模型
        if result_message.structured_output is None:
            return AgentResult(
                success=False,
                error="No structured output in result",
                agent_name=agent_name,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )

        try:
            # 将结构化输出解析为 Pydantic 模型
            parsed_output = response_model.model_validate(result_message.structured_output)

            # 构建成功结果
            result = AgentResult(
                success=True,
                output=parsed_output,
                raw_output=str(result_message.structured_output),
                agent_name=agent_name,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )

            # 保存输出
            _save_agent_output(result, output_folder)

            return result

        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Failed to parse structured output: {str(e)}",
                raw_output=str(result_message.structured_output),
                agent_name=agent_name,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )

    except Exception as e:
        # 外层异常处理：捕获导入错误、网络错误等
        return AgentResult(
            success=False,
            error=f"Agent execution failed: {str(e)}",
            agent_name=agent_name,
            execution_time_ms=int((time.time() - start_time) * 1000)
        )