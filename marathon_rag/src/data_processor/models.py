from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class DocumentNode:
    """文檔樹節點"""
    text: str = ""  # 段落文本
    level: int = 0  # 段落級別（數字越小級別越高）
    parent_index: Optional[int] = None  # 父節點索引
    children_indices: List[int] = field(default_factory=list)  # 子節點索引列表
    node_type: str = "paragraph"  # 節點類型：root, heading, paragraph
    tables: List[Dict[str, Any]] = field(default_factory=list)  # 該節點關聯的表格

@dataclass
class TextChunk:
    """文本分块"""
    chunk_id: str  # 分块ID
    content: str  # 分块内容
    path_titles: List[str]  # 回溯路径上的标题
    source_node_index: int  # 源叶子节点索引
    chunk_index: int  # 在该叶子节点中的分块索引
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据