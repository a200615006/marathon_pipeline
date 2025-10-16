import os
import json
import math
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import load_index_from_storage
from transformers import AutoTokenizer, AutoConfig
from .config import MILVUS_DB_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, MILVUS_STORAGE, IF_IVFPQ
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .utils import save_nodes, load_nodes
import logging

# 基本配置
logging.basicConfig(level=logging.INFO)
log = logging.info  # 或者使用 logger

def init_embed_model():
    """初始化嵌入模型。"""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        cache_folder=None,
        trust_remote_code=True,
        local_files_only=True
    )
    config = AutoConfig.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True, local_files_only=True)
    return config.hidden_size


def calculate_nlist(vector_count):
    """
    根据向量数量动态计算聚类中心数

    Args:
        vector_count: 向量数量

    Returns:
        int: 聚类中心数 nlist
    """
    if vector_count <= 10000:
        num_cluster = max(1, vector_count // 50)  # 确保至少为1
    else:
        num_cluster = int(4 * math.sqrt(vector_count))

    # 限制在合理范围内 [1, 65536]
    num_cluster = max(1, min(num_cluster, 65536))

    log(f"📊 向量数量: {vector_count}, 计算得到聚类中心数 nlist={num_cluster}")
    return num_cluster


def create_milvus_vector_store(overwrite=True, vector_count=None):
    """
    创建 Milvus 向量存储。

    Args:
        overwrite: 是否覆盖现有集合
        vector_count: 向量数量，用于动态计算 nlist（仅在使用 IVF_FLAT 时需要）
    """
    abs_db_path = os.path.abspath(MILVUS_DB_PATH)
    if not os.path.exists(os.path.dirname(abs_db_path)):
        os.makedirs(os.path.dirname(abs_db_path))

    if int(IF_IVFPQ) == 1:
        # 动态计算 nlist
        if vector_count is None:
            # 如果没有提供 vector_count，使用默认值
            nlist = 128
            log(f"⚠️  未提供向量数量，使用默认 nlist={nlist}")
        else:
            nlist = calculate_nlist(vector_count)

        # 动态计算 nprobe（建议为 nlist 的 5-10%）
        nprobe = 10  # 限制最大为64以保证性能

        index_config = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {
                "nlist": nlist
            }
        }

        search_config = {
            "params": {
                "nprobe": nprobe
            }
        }

        log(f"🔧 IVF_FLAT 配置: nlist={nlist}, nprobe={nprobe}")

        return MilvusVectorStore(
            uri=abs_db_path,
            collection_name="rag_collection",
            dim=init_embed_model(),
            overwrite=overwrite,
            index_config=index_config,
            search_config=search_config
        ), index_config, search_config
    else:
        return MilvusVectorStore(
            uri=abs_db_path,
            collection_name="rag_collection",
            dim=init_embed_model(),
            overwrite=overwrite
        ), None, None


def build_index(documents, storage_context, milvus_config=None, reuse_nodes=None):
    """
    构建向量索引。

    Args:
        documents: 文档列表
        storage_context: 存储上下文
        milvus_config: Milvus 配置信息（用于保存）
        reuse_nodes: 预先解析好的节点（可选，避免重复解析）
    """
    qwen_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        tokenizer=qwen_tokenizer.tokenize
    )

    # 如果提供了预解析的节点，直接使用
    if reuse_nodes is not None:
        nodes = reuse_nodes
        log(f"♻️  重用已解析的 {len(nodes)} 个节点")

        # 从节点构建索引
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=Settings.embed_model,
            # 注意：使用预解析节点时，不要设置 store_nodes_override
            show_progress=True
        )
    else:
        # 否则从文档构建
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=Settings.embed_model,
            transformations=[node_parser],
            store_nodes_override=True,
            show_progress=True
        )
        nodes = node_parser.get_nodes_from_documents(documents)

    # 将 milvus 配置附加到 index 对象上，便于后续保存
    if milvus_config:
        index._milvus_config = milvus_config

    return index, nodes


def save_milvus_index(index, persist_dir):
    """保存 Milvus 索引及配置。"""
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    # 保存索引
    index.storage_context.persist(persist_dir=persist_dir)
    log(f"✅ 索引已保存到: {persist_dir}")

    # 保存索引信息（包括 IVF_FLAT 配置）
    index_info = {
        'collection_name': 'rag_collection',
        'milvus_db_path': MILVUS_DB_PATH,
        'embedding_dim': init_embed_model(),
        'total_documents': len(index.docstore.docs),
        'index_type': 'VectorStoreIndex',
        'use_ivf_flat': int(IF_IVFPQ) == 1,
    }

    # 如果使用了 IVF_FLAT，保存配置
    if hasattr(index, '_milvus_config'):
        index_info['milvus_index_config'] = index._milvus_config.get('index_config')
        index_info['milvus_search_config'] = index._milvus_config.get('search_config')
        log(f"✅ IVF_FLAT 配置已保存")

    info_file = persist_path / "index_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(index_info, f, ensure_ascii=False, indent=2)
    log(f"✅ 索引信息已保存到: {info_file}")


def load_milvus_index(persist_dir, milvus_db_path=None):
    """加载 Milvus 索引及配置。"""
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        raise FileNotFoundError(f"❌ 找不到索引目录: {persist_dir}")

    # 读取索引信息
    info_file = persist_path / "index_info.json"
    index_config = None
    search_config = None

    if info_file.exists():
        with open(info_file, 'r', encoding='utf-8') as f:
            index_info = json.load(f)

        milvus_db_path = milvus_db_path or index_info.get('milvus_db_path')
        use_ivf_flat = index_info.get('use_ivf_flat', False)

        # 如果使用了 IVF_FLAT，加载配置
        if use_ivf_flat:
            index_config = index_info.get('milvus_index_config')
            search_config = index_info.get('milvus_search_config')
            if index_config and search_config:
                log(f"✅ 加载 IVF_FLAT 配置: nlist={index_config['params']['nlist']}, "
                      f"nprobe={search_config['params']['nprobe']}")

    # 创建 Milvus 向量存储
    if index_config and search_config:
        milvus_vector_store = MilvusVectorStore(
            uri=milvus_db_path,
            collection_name="rag_collection",
            dim=init_embed_model(),
            overwrite=False,
            index_config=index_config,
            search_config=search_config
        )
    else:
        milvus_vector_store = MilvusVectorStore(
            uri=milvus_db_path,
            collection_name="rag_collection",
            dim=init_embed_model(),
            overwrite=False
        )

    storage_context = StorageContext.from_defaults(
        vector_store=milvus_vector_store,
        persist_dir=persist_dir
    )

    index = load_index_from_storage(
        storage_context=storage_context,
        embed_model=Settings.embed_model
    )

    log(f"✅ 索引已加载")
    return index
