import time
from contextlib import asynccontextmanager
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import asyncio
from advancedMCPHttpToolManager import AdvancedMCPHttpToolManager
from config import MAX_MCP_CALL, EXAM_PORT, OPENAI_API_KEY, OPENAI_API_BASE, MCP_DIRECTORY, X_App_Id, \
    X_App_Key, MAIN_LOG_FILE
from rag_call import RagTool
from req_resp_obj import ChoiceQuestionResponse, ChoiceQuestionRequest
import logging

# 配置日志
logging.basicConfig(
    filename=MAIN_LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)

# 为 RagTool.call 创建专用线程池（限制并发数）
rag_thread_pool = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理器"""
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
    logger.info("=== 应用程序启动 ===")

    yield

    # 关闭时清理
    rag_thread_pool.shutdown(wait=True)
    print("🛑 应用关闭")


app = FastAPI(
    title="创新大赛答题 API 服务",
    description="处理选择题和问答题的 HTTP 服务",
    version="1.0.0",
    lifespan=lifespan
)

async def process_choice_question(question: str, content: str, req_id: int) -> str:
    try:
        result = tool_manager.process_user_query(question, content, req_id)
        if result.code == "1":
            loop = asyncio.get_event_loop()
            rag = await loop.run_in_executor(
                rag_thread_pool,  # 直接使用线程池控制并发
                RagTool.call,
                question,
                content
            )
            return rag["result"]
        else:
            return result.response
    except Exception as e:
        logger.error(f"处理选择题时发生错误: {str(e)}")
        raise Exception(f"查询处理失败: {str(e)}")

@app.post("/api/exam", response_model=ChoiceQuestionResponse)
async def exam(request: ChoiceQuestionRequest):
    """
    主答题接口 - 完全异步处理
    """
    call_start = time.time()

    try:
        logger.info(
            f"收到请求 - segments: {request.segments}, paper: {request.paper}, ID: {request.id}, category: {request.category}")
        logger.info(f"question: {request.question}，content: {request.content}")

        print(f"\n###########################################################################")
        print(
            f"收到请求 - segments: {request.segments}, paper: {request.paper}, ID: {request.id}, category: {request.category}")
        print(f"question: {request.question}，content: {request.content}")

        # 直接调用异步函数
        answer = await process_choice_question(request.question, request.content, request.id)

        # 构建响应
        response = ChoiceQuestionResponse(
            segments=request.segments,
            paper=request.paper,
            id=request.id,
            answer=answer
        )

        cost_time = time.time() - call_start
        print(f"返回答案: {response}")
        print(f"cost={cost_time:.2f}s")
        logger.info(f"请求处理完成 - ID: {request.id}, 耗时: {cost_time:.2f}s")

        return response

    except Exception as e:
        error_msg = f"处理请求时发生错误: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/")
async def root():
    return {
        "message": "MCP工具服务API - 支持本地和HTTP工具",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/concurrency-status")
async def concurrency_status():
    """查看当前 RagTool 并发状态"""
    if hasattr(rag_thread_pool, '_work_queue'):
        queue_size = rag_thread_pool._work_queue.qsize()
        active_count = rag_thread_pool._max_workers - (
                    rag_thread_pool._max_workers - rag_thread_pool._counter._semaphore._value)
    else:
        queue_size = 0
        active_count = 0

    return {
        "max_rag_concurrent": 10,
        "active_rag_calls": active_count,
        "queued_rag_calls": queue_size,
        "available_rag_slots": rag_thread_pool._max_workers - active_count
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=EXAM_PORT,
        log_level="info",
        workers=1,  # 可以增加worker数量来提升并发
        timeout_keep_alive=120
    )