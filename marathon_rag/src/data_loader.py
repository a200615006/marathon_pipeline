from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from llama_index.core import Document
from llama_index.readers.file import UnstructuredReader, PyMuPDFReader
from tqdm import tqdm
from transformers import AutoTokenizer
from .utils import clean_text
from .config import DOCUMENTS_DIR
import logging

# 基本配置
logging.basicConfig(level=logging.INFO)
log = logging.info  # 或者使用 logger
def load_single_file(file_path, file_extractor):
    """加载单个文件。"""
    try:
        ext = Path(file_path).suffix.lower()
        if ext in file_extractor:
            reader = file_extractor[ext]
            log(f'loading: {file_path}')
            docs = reader.load_data(file_path)
            return docs
        return []
    except Exception as e:
        log(f"加载文件 {file_path} 失败: {e}")
        return []

def load_documents_parallel(documents_dir=DOCUMENTS_DIR, file_extractor=None, max_workers=4):
    """并行加载文档。"""
    if file_extractor is None:
        file_extractor = {
            ".docx": UnstructuredReader(),
            ".doc": UnstructuredReader(),
            ".txt": UnstructuredReader(),
            ".md": UnstructuredReader(),
            ".pdf": PyMuPDFReader(),
        }
    all_files = []
    for ext in file_extractor.keys():
        all_files.extend(Path(documents_dir).rglob(f"*{ext}"))
    print("所有文件数量：",len(all_files))
    documents = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single_file, str(f), file_extractor): f for f in all_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="加载文件"):
            docs = future.result()
            documents.extend(docs)
    return documents

def load_documents_serial(documents_dir=DOCUMENTS_DIR, file_extractor=None):
    """串行加载文档。"""
    if file_extractor is None:
        file_extractor = {
            ".docx": UnstructuredReader(),
            ".doc": UnstructuredReader(),
            ".txt": UnstructuredReader(),
            ".md": UnstructuredReader(),
            ".pdf": PyMuPDFReader(),
        }
    all_files = []
    for ext in file_extractor.keys():
        all_files.extend(Path(documents_dir).rglob(f"*{ext}"))
    
    documents = []
    for file_path in tqdm(all_files, desc="加载文件"):
        docs = load_single_file(str(file_path), file_extractor)
        documents.extend(docs)
    
    return documents

def preprocess_long_documents(documents, max_length=100000, overlap=0):
    """预处理超长文档。"""
    processed_docs = []
    for doc in documents:
        text_length = len(doc.text)
        if text_length > max_length:
            log(f"检测到超长文档: {text_length} 字符，进行预切分")
            chunks = []
            start = 0
            chunk_index = 0
            while start < text_length:
                end = min(start + max_length, text_length)
                chunk_text = doc.text[start:end]
                new_metadata = doc.metadata.copy() if doc.metadata else {}
                new_metadata['chunk_index'] = chunk_index
                new_metadata['total_chunks'] = (text_length + max_length - overlap - 1) // (max_length - overlap)
                new_metadata['is_chunked'] = True
                chunks.append(Document(text=chunk_text, metadata=new_metadata))
                start += (max_length - overlap)
                chunk_index += 1
            processed_docs.extend(chunks)
            log(f"  切分为 {len(chunks)} 个块，每块最大 {max_length} 字符，重叠 {overlap} 字符")
        else:
            processed_docs.append(doc)
    return processed_docs