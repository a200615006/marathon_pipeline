import re
import os
import pickle
import json
from pathlib import Path
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import asyncio
from llama_index.core import Settings
import gc
import torch
import logging

# 基本配置
logging.basicConfig(level=logging.INFO)
log = logging.info  # 或者使用 logger
def clean_text(text: str) -> str:
    """清理文本内容。"""
    text = re.sub(r'\\n\\s*\\n+', '\\n\\n', text).strip()
    return text

async def generate_summary_async(text, max_words=30):
    """异步生成摘要。"""
    prompt = f"总结以下文本，不超过{max_words}字，直接回复结果：{text}"
    response = await Settings.llm.acomplete(prompt)
    return response.text.strip()

def generate_summary(text, max_words=30):
    """同步生成摘要。"""
    prompt = f"总结以下文本，不超过{max_words}字，直接回复结果：{text}"
    response = Settings.llm.complete(prompt)
    return response.text.strip()

async def add_summaries_to_nodes_async(nodes_list):
    """异步为节点添加摘要。"""
    tasks = [generate_summary_async(node.text) for node in nodes_list]
    summaries = []
    for future in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="生成节点摘要进度"):
        summary = await future
        summaries.append(summary)
    for node, summary in zip(nodes_list, summaries):
        node.metadata["node_summary"] = summary

def add_summaries_to_nodes(nodes_list):
    """同步为节点添加摘要。"""
    for node in tqdm(nodes_list, desc="生成摘要"):
        summary = generate_summary(node.text)
        node.metadata["node_summary"] = summary

def save_nodes(nodes, save_dir):
    """保存节点数据。"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    pickle_file = save_path / "nodes.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump(nodes, f)
    log(f"Nodes已保存到: {pickle_file}")

def load_nodes(save_dir):
    """加载节点数据。"""
    save_path = Path(save_dir)
    pickle_file = save_path / "nodes.pkl"
    if not pickle_file.exists():
        raise FileNotFoundError(f"❌ 找不到节点文件: {pickle_file}")
    with open(pickle_file, 'rb') as f:
        nodes = pickle.load(f)
    log(f"✅ 已加载 {len(nodes)} 个节点")
    return nodes

def clear_gpu_cache():
    """清理 GPU 缓存。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()