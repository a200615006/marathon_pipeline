#!/usr/bin/env python3
"""
RAG 查询入口 - 使用 FastAPI 部署的查询服务
"""

import os
from pathlib import Path
from src.config import *
from src.llm import SiliconFlowLLM
from src.index_builder import load_milvus_index, init_embed_model
from src.utils import load_nodes
from src.retriever import BM25Retriever, VectorIndexRetriever, QueryFusionRetriever, Qwen3Reranker, SplitNodeRetriever, Qwen3Reranker_vllm
from src.query_engine import create_response_synthesizer_QA, create_response_synthesizer_CHOICE, DynamicQueryEngine
from llama_index.core import Settings
import logging
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
import re

def extract_option(response):
    """
    从LLM响应中提取选项字母（A-Z，a-z）
    优先匹配大写字母，再匹配小写字母；每种类型内先匹配单独字母，再匹配任意字母
    """
    # 第一步：优先匹配单独的大写字母（A-Z）
    match = re.search(r'\b([A-Z])\b', str(response))
    if match:
        return match.group(1)
    
    # 第二步：若第一步失败，匹配任意大写字母（A-Z）
    match = re.search(r'[A-Z]', str(response))
    if match:
        return match.group(0)
    
    # 第三步：若大写字母未找到，匹配单独的小写字母（a-z）
    match = re.search(r'\b([a-z])\b', str(response))
    if match:
        return match.group(1)
    
    # 第四步：若第三步失败，匹配任意小写字母（a-z）
    match = re.search(r'[a-z]', str(response))
    if match:
        return match.group(0)
    
    # 兜底：若所有字母均未找到，返回原始响应字符串
    return str(response)

# 基本配置
logging.basicConfig(level=logging.INFO)
log = logging.info  # 或者使用 logger

app = FastAPI(title="RAG 查询服务", description="基于 Milvus 和 Llama Index 的 RAG 查询 API")

class QueryRequest(BaseModel):
    question: str
    content: str = None

# 全局变量，用于存储加载的组件
index = None
nodes = None
dynamic_query_engine_QA = None
dynamic_query_engine_CHOICE = None
reranker = None
split_retriever = None

def check_database_exists():
    """检查数据库是否已存在"""
    milvus_storage_path = Path(MILVUS_STORAGE)
    saved_nodes_path = Path(SAVED_NODES)
    
    # 检查索引存储目录和节点文件是否存在
    index_exists = milvus_storage_path.exists() and (milvus_storage_path / "index_info.json").exists()
    nodes_exists = saved_nodes_path.exists()
    
    return index_exists and nodes_exists

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理器，用于启动和关闭时执行操作"""
    # 启动逻辑
    log("🔍 RAG 查询系统启动...")

    # 检查数据库是否存在
    if not check_database_exists():
        raise RuntimeError("❌ 数据库不存在，请先构建数据库：python -m src.build_database")

    # 初始化 LLM 和嵌入模型
    log("📝 初始化模型...")
    global index, nodes, dynamic_query_engine_QA, dynamic_query_engine_CHOICE, reranker, split_retriever
    
    llm = SiliconFlowLLM(model=LLM_MODEL, api_key=API_KEY)
    Settings.llm = llm
    dimension = init_embed_model()
    log(f"✅ 嵌入模型维度: {dimension}")
    
    # 加载索引和节点
    log("📂 加载索引...")
    index = load_milvus_index(MILVUS_STORAGE)
    nodes = load_nodes(SAVED_NODES)
    log(f"✅ 加载了 {len(nodes)} 个节点")
    
    # 创建检索器
    log("🔧 创建检索器...")
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=20)
    vector_retriever = index.as_retriever(similarity_top_k=20)
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=30,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
        verbose=True
    )
    if int(IF_VLLMRERANKER):
        reranker = Qwen3Reranker_vllm(
            model=RERANKER_MODEL, top_n=10, batch_size=BATCH_SIZE,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.8, instruction='给定一个数据库搜索查询，检索有助于回答该查询的相关段落'
        )
    else:
        reranker = Qwen3Reranker(model=RERANKER_MODEL, top_n=10, batch_size=BATCH_SIZE,
                                 use_kv_cache=True,
                                 max_kv_cache_size=10,  # 每10个批次清理一次
                                 enable_gradient_checkpointing=False,
                                 instruction='给定一个数据库搜索查询，检索有助于回答该查询的相关段落'
                                 )

    # 创建查询引擎
    response_synthesizer_QA = create_response_synthesizer_QA()
    response_synthesizer_CHOICE = create_response_synthesizer_CHOICE()

    split_retriever = SplitNodeRetriever(hybrid_retriever, chunk_size=512, overlap_ratio=0)

    dynamic_query_engine_QA = DynamicQueryEngine(
        retriever=split_retriever,
        response_synthesizer=response_synthesizer_QA,
        reranker=reranker,
        keep_top_k=5,
        use_parent_nodes=True,
        reorder=True
    )
    dynamic_query_engine_CHOICE = DynamicQueryEngine(
        retriever=split_retriever,
        response_synthesizer=response_synthesizer_CHOICE,
        reranker=reranker,
        keep_top_k=5,
        use_parent_nodes=True,
        reorder=True
    )

    log("🎉 RAG 系统准备就绪！")

    yield  # 应用运行期间

    # 关闭逻辑（如果需要清理资源）
    log("🛑 RAG 查询系统关闭...")

# 将 lifespan 绑定到 app
app = FastAPI(lifespan=lifespan, title="RAG 查询服务", description="基于 Milvus 和 Llama Index 的 RAG 查询 API")

@app.post("/query", response_model=dict)
async def query_endpoint(request: QueryRequest = Body(...)):
    """处理 RAG 查询请求"""
    try:
        log("🔍 正在查询...")
        if request.content is not None:
            response = extract_option(dynamic_query_engine_CHOICE.query(request.question + request.content))
        else:
            response = dynamic_query_engine_QA.query(request.question)
        
        return {"answer": str(response)}
    
    except Exception as e:
        log(f"❌ 查询出错: {e}")
        raise HTTPException(status_code=500, detail=f"查询出错: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=28080)
