import time
from contextlib import asynccontextmanager

import os
from datetime import datetime, date
from fastapi import FastAPI, HTTPException, Query, Body
import uvicorn
from advancedMCPHttpToolManager import AdvancedMCPHttpToolManager
from config import MAX_MCP_CALL, EXAM_PORT, TEST_PORT, OPENAI_API_KEY, OPENAI_API_BASE, MCP_DIRECTORY, X_App_Id, \
    X_App_Key, MAIN_LOG_FILE
from rag_call import RagTool
from req_resp_obj import ToolResponse, ToolRequest, QueryResponse, UserQuery, ChoiceQuestionResponse, ChoiceQuestionRequest, QAQuestionRequest, QAQuestionResponse
import logging


# 配置日志
logging.basicConfig(
    filename=MAIN_LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)
# 测试日志
logger.info("=== 应用程序启动 ===")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理器"""
    # 启动时初始化
    global tool_manager
    tool_manager = AdvancedMCPHttpToolManager(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        tools_directory=MCP_DIRECTORY,
        max_iterations=MAX_MCP_CALL,
        headers={
            "X-App-Id": X_App_Id,
            "X-App-Key": X_App_Key,
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
        result = tool_manager.process_user_query(question,  content)
        if result.code == "1":
            rag = RagTool.call(question, content)
            print(f"rag_result={rag}")
            result = rag["result"]
            return result
        else:
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
        call_start = time.time()
        print(f"\n###########################################################################")
        logger.info(
            f"收到请求 - segments: {request.segments}, paper: {request.paper}, ID: {request.id}, category: {request.category}")
        logger.info(f"question: {request.question}，content: {request.content}")

        print(f"收到请求 - segments: {request.segments}, paper: {request.paper}, ID: {request.id}, category: {request.category}")
        print(f"question: {request.question}，content: {request.content}")

        answer = process_choice_question(request.question, request.content)

        # 构建响应
        response = ChoiceQuestionResponse(
            segments=request.segments,
            paper=request.paper,
            id=request.id,
            answer=answer
        )

        print(f"返回答案: {response}")
        print(f"cost={time.time()-call_start:.2f}s")
        return response

    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")


    """健康检查端点"""
@app.get("/health")
async def health_check():
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
        port=EXAM_PORT,
        log_level="info",
        workers=1,
        timeout_keep_alive=120  # 设置超时时间为120秒
    )