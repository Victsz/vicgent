# LLMUtil 使用指南

## 概述

`LLMUtil` 模块提供了一个标准化的 Anthropic 模型创建工具，基于 `src/vicgent/core/extractor.py` 中的 `tllm` 参考实现。

## 安装与导入

确保项目依赖已安装，然后导入模块：

```python
from vicgent.util.LLMUtil import create_model_anthropic
```

## 核心函数

### `create_model_anthropic`

创建标准化的 Anthropic LLM 模型实例。

**参数：**
- `model_name` (str): 模型名称（例如："Pro/deepseek-ai/DeepSeek-V3"）
- `base_url` (Optional[str]): API 端点的基础 URL，如果为 None 则使用环境变量 ANTHROPIC_BASE_URL
- `temperature` (float): 采样温度（0 表示确定性输出）
- `api_key_env_var` (str): API 密钥的环境变量名称，默认为 "API_KEY"
- `**kwargs`: 传递给 `init_chat_model` 的额外参数

**返回值：**
配置好的 Anthropic 聊天模型实例

## 环境变量配置

在使用前，请确保设置以下环境变量：

```bash
# 在 .agent.env 或系统环境变量中设置
ANTHROPIC_BASE_URL=your_anthropic_base_url
API_KEY=your_api_key
TMODEL=Pro/deepseek-ai/DeepSeek-V3  # 文本模型默认值
VMODEL=Qwen/Qwen2.5-VL-72B-Instruct  # 视觉模型默认值
```

## 使用示例

### 基本用法

```python
from vicgent.util.LLMUtil import create_model_anthropic

# 使用环境变量中的模型配置
tllm = create_model_anthropic(
    model_name=os.getenv("TMODEL", "Pro/deepseek-ai/DeepSeek-V3")
)

# 测试模型连接
response = tllm.invoke("Hello, are you ready?")
print(response.content)
```

### 自定义参数

```python
# 显式指定所有参数
custom_model = create_model_anthropic(
    model_name="Pro/deepseek-ai/DeepSeek-V3",
    base_url="https://api.anthropic.com",
    temperature=0.1,
    api_key_env_var="ANTHROPIC_API_KEY"
)
```

### 在 extractor.py 中的替代用法

原代码：
```python
tllm = init_chat_model(
    model=TMODEL,
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
    model_provider="anthropic",
    temperature=0,
)
```

新代码：
```python
from vicgent.util.LLMUtil import create_model_anthropic

tllm = create_model_anthropic(
    model_name=TMODEL,
    temperature=0
)
```

## 最佳实践

1. **环境变量管理**: 始终通过环境变量管理敏感信息
2. **错误处理**: 在使用模型前检查环境变量是否已设置
3. **温度设置**: 工具调用场景使用 temperature=0 确保结构化输出
4. **模型选择**: 根据任务类型选择合适的模型（文本处理 vs 视觉处理）

## 故障排除

**常见问题：**
- 环境变量未设置：确保 `.agent.env` 文件存在且包含必要的配置
- API 密钥错误：检查 `API_KEY` 环境变量是否正确设置
- 网络连接问题：验证 `ANTHROPIC_BASE_URL` 可访问

**调试提示：**
```python
import os
print(f"ANTHROPIC_BASE_URL: {os.getenv('ANTHROPIC_BASE_URL')}")
print(f"API_KEY present: {bool(os.getenv('API_KEY'))}")
```

## 相关文件

- `src/vicgent/util/LLMUtil.py`: 工具实现
- `src/vicgent/core/extractor.py`: 参考实现
- `.agent.env`: 环境配置文件