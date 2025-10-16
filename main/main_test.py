from contextlib import asynccontextmanager

import os
from datetime import datetime, date
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from advancedMCPHttpToolManager import AdvancedMCPHttpToolManager
from req_resp_obj import ToolResponse, ToolRequest, QueryResponse, UserQuery, ChoiceQuestionResponse, ChoiceQuestionRequest, QAQuestionRequest, QAQuestionResponse
import logging


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理器"""
    # 启动时初始化
    global tool_manager
    tool_manager = AdvancedMCPHttpToolManager(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_URL"),
        tools_directory="mcp_tools",
        max_iterations=5,
        headers={
            "X-App-Id": os.getenv("your_app_id"),
            "X-App-Key": os.getenv("your_app_key"),
            "Content-Type": "application/json"
        }
    )
    print("🚀 MCP工具管理器初始化完成")

    yield  # 应用运行期间

    # 关闭时清理（如果需要）
    print("🛑 应用关闭")


# FastAPI 应用
app = FastAPI(
    title="创新大赛答题 API 服务",
    description="处理选择题和问答题的 HTTP 服务",
    version="1.0.0",
    lifespan=lifespan  # 使用 lifespan 事件处理器
)


def process_choice_question(question: str, content: str) -> str:
    """
    处理选择题
    根据问题内容和选项分析正确答案
    """

    try:
        result = tool_manager.process_user_query(question,  content or "")
        if result.code == "1":
            pass
            # todo rag
        print(result)
        print(result.response)
        return result.response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")


@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "MCP工具服务API - 支持本地和HTTP工具",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/exam", response_model=ChoiceQuestionResponse)
async def exam(request: ChoiceQuestionRequest):
    """
    主答题接口
    接收问题并返回答案
    """
    try:
        logger.info(
            f"收到请求 - segments: {request.segments}, paper: {request.paper}, ID: {request.id}, category: {request.category}")
        logger.info(f"question: {request.question}，content: {request.content}")

        answer = process_choice_question(request.question, request.content or "")

        # 构建响应
        response = ChoiceQuestionResponse(
            segments=request.segments,
            paper=request.paper,
            id=request.id,
            answer=answer
        )

        logger.info(f"返回答案: {response}")
        return response

    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "tools_loaded": len(tool_manager.tools) if tool_manager else 0
    }


if __name__ == "__main__":
    # 启动FastAPI服务器
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=20001,
        log_level="info",
        workers=1
    )