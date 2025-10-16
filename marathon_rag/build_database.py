#!/usr/bin/env python3
"""
数据库构建入口 - 用于首次运行时的数据库建立
"""

import os
from pathlib import Path
from src.config import *
from src.llm import SiliconFlowLLM
from src.data_loader import load_documents_parallel, preprocess_long_documents,load_documents_serial
from src.index_builder import create_milvus_vector_store, build_index, save_milvus_index, init_embed_model
from src.utils import clean_text, save_nodes
from llama_index.core import Settings, StorageContext, Document
from llama_index.core.node_parser import SentenceSplitter
from transformers import AutoTokenizer, AutoConfig
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

def build_database():
    """构建数据库"""
    log("🚀 开始构建数据库...")
    
    # 检查数据库是否已存在
    if check_database_exists():
        log("⚠️  数据库已存在，如需重新构建请删除以下目录：")
        log(f"   - {MILVUS_STORAGE}")
        log(f"   - {SAVED_NODES}")
        log("\n如需使用现有数据库，请运行: python -m src.main")
        return
    
    # 初始化 LLM 和嵌入模型
    log("📝 初始化模型...")
    llm = SiliconFlowLLM(model=LLM_MODEL, api_key=API_KEY)
    Settings.llm = llm
    dimension = init_embed_model()
    log(f"✅ 嵌入模型维度: {dimension}")
    
    # 加载文档
    log("📚 加载文档...")
    documents = load_documents_parallel()
    log(f"✅ 加载了 {len(documents)} 个文档")
    
    # 清理和预处理文档
    log("🧹 清理和预处理文档...")
    cleaned_documents = [Document(text=clean_text(doc.text), metadata=doc.metadata) for doc in documents]
    documents = preprocess_long_documents(cleaned_documents, max_length=MAX_LENGTH, overlap=0)
    log(f"✅ 预处理后文档数量: {len(documents)}")
    
    # 构建向量存储和索引
    log("🔧 预解析文档...")
    # 只解析一次，获取节点和数量
    qwen_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        tokenizer=qwen_tokenizer.tokenize
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    vector_count = len(nodes)
    log(f"📊 解析完成，生成 {vector_count} 个节点")

    # 检查节点ID是否重复
    node_ids = [node.node_id for node in nodes]
    unique_ids = set(node_ids)
    if len(node_ids) != len(unique_ids):
        log(f"⚠️  警告：发现重复节点ID！总数={len(node_ids)}, 唯一={len(unique_ids)}")
    else:
        log(f"✅ 节点ID检查通过，无重复")

    log("🔧 构建向量存储...")
    # 使用向量数量创建 vector store
    vector_store, index_config, search_config = create_milvus_vector_store(
        overwrite=True,
        vector_count=vector_count
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    log("🏗️  构建索引...")
    # 传递配置信息和已解析的节点
    milvus_config = {
        'index_config': index_config,
        'search_config': search_config
    }
    index, nodes = build_index(
        documents,
        storage_context,
        milvus_config,
        reuse_nodes=nodes  # 重用已解析的节点
    )
    log(f"✅ 构建了 {len(nodes)} 个节点")

    log("💾 保存索引...")
    save_milvus_index(index, MILVUS_STORAGE)
    save_nodes(nodes, SAVED_NODES)
    log("🎉 数据库构建完成！")
    log(f"📁 索引保存在: {MILVUS_STORAGE}")
    log(f"📁 节点保存在: {SAVED_NODES}")
    log("\n现在可以运行 RAG 查询: python main.py")

if __name__ == "__main__":
    build_database()