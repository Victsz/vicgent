#!/usr/bin/env python3
"""
Claude 无头包装器

用于通过子进程调用 Claude CLI，执行指定查询并返回结构化 JSON 响应。
"""

import subprocess 
import json
import os
import argparse
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClaudeHeadlessWrapper:
    """Claude 无头包装器类"""

    def __init__(
        self,
        working_dir: str = "/home/victor/workspace_local/agent_services/",
        settings_file: str = "/home/victor/workspace_local/agent_services/.claude/settings.local.json"
    ):
        """
        初始化 Claude 无头包装器

        Args:
            working_dir: 工作目录路径
            settings_file: Claude 设置文件路径
        """
        self.working_dir = Path(working_dir)
        self.settings_file = Path(settings_file)
        self.session_id = None
        # 验证路径存在性 - fail fast 原则
        if not self.working_dir.exists():
            raise FileNotFoundError(f"工作目录不存在: {self.working_dir}")

        if not self.settings_file.exists():
            raise FileNotFoundError(f"设置文件不存在: {self.settings_file}")

    def _save_output_to_log(self, stdout: str, stderr: str, query: str) -> str:
        """
        将输出保存到日志文件

        Args:
            stdout: 标准输出
            stderr: 标准错误输出
            query: 执行的查询

        Returns:
            str: 日志文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"claude_output_{timestamp}.log"
        log_path = self.working_dir / log_filename

        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Claude CLI 执行日志 ===\n")
            f.write(f"执行时间: {datetime.now().isoformat()}\n")
            f.write(f"查询: {query}\n")
            f.write(f"工作目录: {self.working_dir}\n")
            f.write(f"设置文件: {self.settings_file}\n")
            f.write(f"\n=== 标准输出 ===\n")
            f.write(stdout)
            if stderr:
                f.write(f"\n=== 标准错误输出 ===\n")
                f.write(stderr)
            f.write(f"\n=== 日志结束 ===\n")

        logger.info(f"输出已保存到日志文件: {log_path}")
        return str(log_path)
    def run_query_raw(self,
        query: str,
        cc:str = "claude",
        output_format: str = "json",
        json_schema:dict = None,
        system_prompt:str = "",
        session_id:str = None
    ):
        """
        执行 Claude 查询

        Args:
            query: 要执行的查询字符串
            output_format: 输出格式，默认为 json
            system_prompt: 系统提示词

        Returns:
            CompletedProcess
        """
        try:
            json_schema = {
                "type": "object",
                "properties": {
                    "is_success": {"type": "boolean"},
                    "message": {"type": "string"}
                },
                "required": ["is_success", "message"]
            } if json_schema is None else json_schema


            # 构建 Claude 命令 - 使用列表形式确保参数正确处理空格
            cmd = cc.split(' ')
            if session_id:
                cmd.extend(['-r',f'{session_id}'])
            args = [
                "--verbose",
                "-p", query,
                "--output-format", output_format,
                "--append-system-prompt", system_prompt,
                "--json-schema", json.dumps(json_schema),
                "--settings", str(self.settings_file)
            ]
            cmd.extend(args) 

            # 改进日志输出，显示命令结构而不是简单连接
            logger.info("执行命令:")
            logger.info(f"  程序: {cmd[0]}")
            logger.info(f"  查询参数: {query}")
            logger.info(f"  输出格式: {output_format}")
            logger.info(f"  系统提示: {system_prompt}")
            logger.info(f"  设置文件: {self.settings_file}")
            logger.info(f"工作目录: {self.working_dir}")

            # 在指定工作目录中执行命令
            result = subprocess.run(
                cmd,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=3000  # 50分钟超时
            )
        except Exception as ex:
            logger.error(f"执行查询时发生错误: {ex}")
            raise

        return result
    @staticmethod
    def extract_response_data(re:subprocess.CompletedProcess):
        """
        从查询响应中提取关键数据
        
        Args:
            re: 查询响应对象，包含stdout属性
            
        Returns:
            tuple: (result, session_id, structured_output)
                如果提取失败，对应值为None
        """
        result = None
        session_id = None
        structured_output = None
        
        try:
            # 解析JSON响应
            import json
            re_j = json.loads(re.stdout)
            
            # 保存原始响应到文件（调试用）
            with open('/tmp/re.json', 'w') as f:
                json.dump(re_j, f, indent=2, ensure_ascii=False)
            
            # 检查响应结构
            if isinstance(re_j, list) and len(re_j) > 0:
                task_re = re_j[-1]
                
                # 提取result
                if isinstance(task_re, dict) and "result" in task_re:
                    result = task_re["result"]
                    print(f"提取到result: {result}")
                
                # 提取session_id
                if isinstance(task_re, dict) and "session_id" in task_re:
                    session_id = task_re["session_id"]
                    print(f"提取到session_id: {session_id}")
                
                # 提取structured_output
                if isinstance(task_re, dict) and "structured_output" in task_re:
                    structured_output = task_re["structured_output"]
                    print(f"提取到structured_output: {structured_output}")
            
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
        except Exception as e:
            print(f"提取数据时出错: {e}")
        
        return result, session_id, structured_output

    def run_query(
        self,
        query: str,
        cc:str = "ccr code",
        output_format: str = "json",
        system_prompt: str = "请以 {issuccess: bool, output: string, msg:string[len<50] } 的 JSON 格式回复"
    ) -> Dict[str, Any]:
        """
        执行 Claude 查询

        Args:
            query: 要执行的查询字符串
            output_format: 输出格式，默认为 json
            system_prompt: 系统提示词

        Returns:
            Dict[str, Any]: 包含成功状态和日志路径的响应
        """
        try:
            # 直接调用 run_query_raw
            result = self.run_query_raw(
                query=query,
                cc=cc,
                output_format=output_format,
                system_prompt=system_prompt
            )

            # 保存输出到日志文件
            log_path = self._save_output_to_log(result.stdout, result.stderr, query)

            # 检查命令执行状态
            if result.returncode != 0:
                logger.error(f"Claude 命令执行失败，返回码: {result.returncode}")
                return {
                    "issuccess": False,
                    "log_path": log_path,
                    "msg": f"命令执行失败，返回码: {result.returncode}"
                }

            # 成功执行，返回状态和日志路径
            logger.info("Claude 命令执行成功")
            return {
                "issuccess": True,
                "log_path": log_path,
                "msg": "命令执行成功"
            }

        except subprocess.TimeoutExpired:
            logger.error("命令执行超时")
            # 创建超时日志
            log_path = self._save_output_to_log("", "命令执行超时", query)
            return {
                "issuccess": False,
                "log_path": log_path,
                "msg": "命令执行超时"
            }
        except Exception as e:
            logger.error(f"执行查询时发生错误: {e}")
            # 创建错误日志
            log_path = self._save_output_to_log("", f"执行错误: {str(e)}", query)
            return {
                "issuccess": False,
                "log_path": log_path,
                "msg": f"执行错误: {str(e)[:50]}"
            }

    def run_merge_paper(self, paper_path: str) -> Dict[str, Any]:
        """
        执行合并论文的特定查询

        Args:
            paper_path: 论文路径

        Returns:
            Dict[str, Any]: 执行结果
        """
        # 确保路径被正确处理，特别是包含空格的路径
        # 使用引号包围路径以确保 Claude CLI 正确解析
        safe_paper_path = f'"{paper_path}"' if ' ' in paper_path else paper_path
        query = f"run /merge-paper {safe_paper_path};"
        logger.info(f"构建查询: {query}")
        return self.run_query(query)


def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description="Claude 无头包装器 - 执行合并论文查询")
    parser.add_argument(
        "paper_path",
        help="要处理的论文路径"
    )
    parser.add_argument(
        "--working-dir",
        default="/home/victor/workspace_local/agent_services/",
        help="工作目录路径 (默认: /home/victor/workspace_local/agent_services/)"
    )
    parser.add_argument(
        "--settings-file",
        default="/home/victor/workspace_local/agent_services/.claude/settings.local.json",
        help="Claude 设置文件路径 (默认: /home/victor/workspace_local/agent_services/.claude/settings.local.json)"
    )

    args = parser.parse_args()

    try:
        # 创建包装器实例
        wrapper = ClaudeHeadlessWrapper(
            working_dir=args.working_dir,
            settings_file=args.settings_file
        )

        # 执行查询
        result = wrapper.run_merge_paper(args.paper_path)

        print("执行结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"错误: {e}")
        exit(1)


if __name__ == "__main__":
    main()