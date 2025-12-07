# VicGent Extractor 演示

本目录包含演示程序，用于展示 `extractor.py` 中两种图版本的返回差异。

## 文件说明

- `demo_extractor.py`: 主要演示文件，展示两种图版本的返回差异
- `run_demo.py`: 简化的运行脚本，包含环境检查
- `README.md`: 本说明文件

## 演示内容

### 1. 带 RunnableLambda(create_final_reponse) 的版本
- 返回简化的消息格式
- 适合直接向用户展示结果
- 文件路径信息嵌入在消息文本中

### 2. 不带 RunnableLambda(create_final_reponse) 的版本
- 返回完整的状态对象
- 适合程序进一步处理
- 提供结构化的数据访问

## 运行方法

### 方法 1: 使用运行脚本（推荐）
```bash
cd src/vicgent/demo
python run_demo.py
```

### 方法 2: 直接运行演示
```bash
cd src/vicgent/demo
python demo_extractor.py
```

## 环境要求

确保以下环境变量已设置：
- `ANTHROPIC_BASE_URL`: LLM API 端点
- `API_KEY`: 认证密钥
- `VMODEL`: 视觉模型
- `TMODEL`: 文本模型

## 输入文件

演示使用以下输入文件：
```
/home/victor/workspace_local/agent_services/AgentDocsSpace/extract_table/input/processed/2021F_PPMT_RevenueIP.png
```

请确保该文件存在，或者修改代码中的文件路径。

## 输出示例

### 带 create_final_reponse 的版本返回：
```python
{
    "messages": [
        AIMessage(content="Table saved to /path/to/file.md content\n\n| Header1 | Header2 |\n|--------|--------|")
    ]
}
```

### 不带 create_final_reponse 的版本返回：
```python
{
    "messages": [...完整消息历史...],
    "original_file": FileResponse(name="...", path="...", description="..."),
    "final_response": FileResponse(name="...", path="...", description="..."),
    "table_str": "| Header1 | Header2 |\n|--------|--------|"
}
```

## 使用建议

- 如果需要直接向用户展示结果，使用带 `create_final_reponse` 的版本
- 如果需要程序化处理结果，使用不带 `create_final_reponse` 的版本
- 可以根据具体需求选择合适的版本