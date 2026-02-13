"""
Test for vflows.core.agent_base.run_agent_structured
"""
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from vflows.core.agent_base import run_agent_structured
from pydantic import BaseModel, Field


# 加载 cc-sdk.env (override=True 确保覆盖已存在的环境变量)
env_path = Path("cc-sdk.env")
load_dotenv(env_path, override=True)


class TestResponse(BaseModel):
    """Test response model"""
    message: str = Field(description="A greeting message")


async def main():
    """Test run_agent_structured with a simple prompt"""
    print("=" * 50)
    print("Testing vflows.core.agent_base.run_agent_structured")
    print("=" * 50)
    print(f"ANTHROPIC_BASE_URL: {os.getenv('ANTHROPIC_BASE_URL')}")
    print(f"MODEL: {os.getenv('MODEL')}")

    result = await run_agent_structured(
        agent_name="test_agent",
        prompt="HI!",
        model=os.getenv("MODEL", "claude-sonnet-4-20250514"),
        response_model=TestResponse,
        tools=None,
        timeout_seconds=60,
        debug=True,
    )

    print("\n" + "=" * 50)
    print("RESULT:")
    print("=" * 50)
    print(f"Success: {result.success}")
    print(f"Execution time: {result.execution_time_ms}ms")

    if result.success and result.output:
        print(f"Output type: {type(result.output).__name__}")
        print(f"Message: {result.output.message}")
    else:
        print(f"Error: {result.error}")

    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
