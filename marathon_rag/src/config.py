import os
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"

def load_config():
    """加载 YAML 配置文件。"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# 导出常用配置
EMBEDDING_MODEL = CONFIG['models']['embedding']
RERANKER_MODEL = CONFIG['models']['reranker']
LLM_MODEL = CONFIG['models']['llm']
API_KEY = CONFIG['api']['siliconflow_key']
DOCUMENTS_DIR = CONFIG['paths']['documents_dir']
MILVUS_DB_PATH = os.path.join(CONFIG['paths']['milvus_dir'], CONFIG['paths']['milvus_db'])
SAVED_NODES = CONFIG['paths']['saved_nodes']
MILVUS_STORAGE = CONFIG['paths']['milvus_storage']
LOGS_PATH = CONFIG['paths']['logs']
PDF_MODEL_PATH = CONFIG['paths']['pdf_model_path']
CHUNK_SIZE = CONFIG['parameters']['chunk_size']
CHUNK_OVERLAP = CONFIG['parameters']['chunk_overlap']
MAX_TOKENS = CONFIG['parameters']['max_tokens']
TEMPERATURE = CONFIG['parameters']['temperature']
BATCH_SIZE = CONFIG['parameters']['batch_size']
MAX_LENGTH = CONFIG['parameters']['max_length']
IF_IVFPQ = CONFIG['parameters']['if_ivfpq']
IF_VLLMRERANKER = CONFIG['parameters']['if_vllmReranker']
