
from __future__ import annotations
from graph_app import full_graph_with_input
from langserve import add_routes
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP
# 创建一个 FastAPI 应用实例
api = FastAPI(
    title="我的 LangGraph 服务",
    version="1.0",
    description="使用 LangServe 部署的简单 LangGraph 示例"
)

# 将我们的 LangGraph 应用添加到 API 路由
add_routes(
    api,
    full_graph_with_input,
    path="/g",
)

# mcp = FastApiMCP(api)

# # Mount the MCP server directly to your FastAPI app
# mcp.mount_http()

# 命令行运行指南（可选，但推荐）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8887)
    