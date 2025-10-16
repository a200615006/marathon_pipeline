#!/usr/bin/env python3
"""
RAG 查询入口 - 用于数据库建立后的查询操作
"""

import os
from pathlib import Path
from src.config import *
from src.llm import SiliconFlowLLM
from src.index_builder import load_milvus_index, init_embed_model
from src.utils import load_nodes
from src.retriever import BM25Retriever, VectorIndexRetriever, QueryFusionRetriever, Qwen3Reranker, SplitNodeRetriever,Qwen3Reranker_vllm
from src.query_engine import create_response_synthesizer_QA,create_response_synthesizer_CHOICE, DynamicQueryEngine
from llama_index.core import Settings
import logging

# 基本配置
logging.basicConfig(level=logging.INFO)
log = logging.info  # 或者使用 logger
def check_database_exists():
    """检查数据库是否已存在"""
    milvus_storage_path = Path(MILVUS_STORAGE)
    saved_nodes_path = Path(SAVED_NODES)
    
    # 检查索引存储目录和节点文件是否存在
    index_exists = milvus_storage_path.exists() and (milvus_storage_path / "index_info.json").exists()
    nodes_exists = saved_nodes_path.exists()
    
    return index_exists and nodes_exists

def main():
    """主函数 - RAG 查询入口"""
    log("🔍 RAG 查询系统启动...")
    
    # 检查数据库是否存在
    if not check_database_exists():
        log("❌ 数据库不存在，请先构建数据库：")
        log("   python -m src.build_database")
        log("\n或者运行以下命令一次性构建并查询：")
        log("   python -m src.build_and_query")
        return
    
    # 初始化 LLM 和嵌入模型
    log("📝 初始化模型...")
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

    def query_engine(query: str, content: str = None):
        if content is not None:
            response = dynamic_query_engine_CHOICE.query(query + content)
        else:
            response = dynamic_query_engine_QA.query(query)

        return response
    # 交互式查询循环
    while True:
        try:
            query = input("\n📝 查询: ").strip()
            if query.lower() in ['quit', 'exit', '退出']:
                log("👋 再见！")
                break
            
            if not query:
                continue
            
            log("🔍 正在查询...")
            response = query_engine(query)
            log(f"\n📄 回答: {response}")
            
        except KeyboardInterrupt:
            log("\n👋 再见！")
            break
        except Exception as e:
            log(f"❌ 查询出错: {e}")

if __name__ == "__main__":
    main()