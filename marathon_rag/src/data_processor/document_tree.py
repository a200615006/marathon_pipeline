from typing import List, Optional, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 從 chunking 模組導入優化函式
from .chunking import optimize_chunks_after_splitting
# 從 models 模組導入數據類
from .models import DocumentNode, TextChunk
import logging

# 基本配置
logging.basicConfig(level=logging.INFO)
log = logging.info  # 或者使用 logger
class DocumentTree:
    """文檔樹"""
    def __init__(self):
        self.nodes: List[DocumentNode] = []
        self.current_index: int = 0
        self._init_root()

    def _init_root(self):
        """初始化根節點"""
        root = DocumentNode(
            text="ROOT",
            level=0,  # 根節點級別為0
            parent_index=None,
            node_type="root"
        )
        self.nodes.append(root)
        self.current_index = 0

    def add_paragraph(self, text: str, level: int, node_type: str = "paragraph"):
        """添加段落到文檔樹"""
        # log(f"Adding: {node_type} level={level}, text='{text[:30]}...', current_index={self.current_index}, current_level={self.nodes[self.current_index].level}")

        if node_type == "heading":
            # 處理標題：找到合適的父節點
            parent_index = self._find_parent_for_heading(level)

            # 創建新的標題節點
            new_node = DocumentNode(
                text=text,
                level=level,
                parent_index=parent_index,
                node_type=node_type
            )

            new_index = len(self.nodes)
            self.nodes.append(new_node)

            # 更新父節點的子節點列表
            self.nodes[parent_index].children_indices.append(new_index)

            # 更新當前節點為新創建的標題節點
            self.current_index = new_index

            # log(f"  Created heading node {new_index}, parent={parent_index}")

        else:  # paragraph
            # 處理段落：添加到當前節點（通常是最近的標題）
            if self.nodes[self.current_index].text and self.nodes[self.current_index].node_type == "paragraph":
                # 如果當前節點已經是段落且有內容，合併文本
                self.nodes[self.current_index].text += "\n" + text
            elif self.nodes[self.current_index].node_type == "heading":
                # 如果當前節點是標題，創建新的段落節點作為其子節點
                new_node = DocumentNode(
                    text=text,
                    level=999,  # 段落級別設為最大
                    parent_index=self.current_index,
                    node_type="paragraph"
                )

                new_index = len(self.nodes)
                self.nodes.append(new_node)

                # 更新父節點的子節點列表
                self.nodes[self.current_index].children_indices.append(new_index)

                # log(f"  Created paragraph node {new_index} under heading {self.current_index}")
            else:
                # 其他情況，更新當前節點的文本
                if self.nodes[self.current_index].text:
                    self.nodes[self.current_index].text += "\n" + text
                else:
                    self.nodes[self.current_index].text = text

    def _find_parent_for_heading(self, level: int) -> int:
        """為標題找到合適的父節點"""
        current_idx = self.current_index

        # 從當前節點開始向上查找
        while current_idx is not None:
            current_node = self.nodes[current_idx]

            # 如果當前節點是根節點，直接返回
            if current_node.node_type == "root":
                return current_idx

            # 如果當前節點是標題且級別比新標題高（數字更小），則作為父節點
            if current_node.node_type == "heading" and current_node.level < level:
                return current_idx

            # 繼續向上查找
            current_idx = current_node.parent_index

        # 如果沒找到合適的父節點，返回根節點
        return 0

    def add_table_to_current_node(self, table_data: Dict[str, Any]):
        """將表格添加到當前節點"""
        self.nodes[self.current_index].tables.append(table_data)


    def get_leaf_nodes(self) -> List[int]:
        """獲取所有葉子節點的索引"""
        leaf_nodes = []
        for i, node in enumerate(self.nodes):
            if not node.children_indices and node.node_type != "root":
                leaf_nodes.append(i)
        return leaf_nodes


    def get_path_to_root(self, node_index: int) -> List[int]:
        """獲取從指定節點到根節點的路徑"""
        path = []
        current_idx = node_index

        while current_idx is not None:
            path.append(current_idx)
            current_idx = self.nodes[current_idx].parent_index

        return path  # 从叶子节点到根节点的路径


    def get_path_titles(self, node_index: int) -> List[str]:
        """獲取從根節點到指定節點路徑上的所有標題"""
        path = self.get_path_to_root(node_index)
        path.reverse()  # 反转，从根节点到叶子节点

        titles = []
        for idx in path:
            node = self.nodes[idx]
            if node.node_type == "heading" and node.text.strip():
                titles.append(node.text.strip())

        return titles

    def create_chunks_for_leaf_node(self, leaf_node_index: int,
                              max_chunk_size: int = 4096,
                              min_overlap: int = 200,
                              max_overlap: int = 400) -> List[TextChunk]:
      """为叶子节点创建文本分块（使用后处理优化）"""
      leaf_node = self.nodes[leaf_node_index]

      if not leaf_node.text or not leaf_node.text.strip():
          return []

      # 获取路径上的标题
      path_titles = self.get_path_titles(leaf_node_index)

      # 使用原有的分割器
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=max_chunk_size,
          chunk_overlap=min_overlap,
          length_function=len,
          separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";"]
      )

      # 分割文本
      raw_chunks = text_splitter.split_text(leaf_node.text)
      # log(raw_chunks)

      # 后处理优化
      optimized_chunks = optimize_chunks_after_splitting(
          raw_chunks,
          min_chunk_size=100,  # 小于100字符的分块需要合并
          max_overlap=max_overlap      # 最大重叠长度
      )

      # 创建TextChunk对象
      chunks = []
      for i, chunk_text in enumerate(optimized_chunks):
          if hasattr(chunk_text, 'content'):
              # 已经是优化后的TextChunk对象
              chunks.append(chunk_text)
          else:
              # 构建完整内容
              if path_titles:
                  title_text = " > ".join(path_titles)
                  full_content = f"{title_text}\n\n{chunk_text}"
              else:
                  full_content = chunk_text

              chunk_id = f"node_{leaf_node_index}_chunk_{i}"

              chunk = TextChunk(
                  chunk_id=chunk_id,
                  content=full_content,
                  path_titles=path_titles,
                  source_node_index=leaf_node_index,
                  chunk_index=i,
                  metadata={
                      'original_text_length': len(leaf_node.text),
                      'chunk_text_length': len(chunk_text),
                      'full_content_length': len(full_content),
                      'node_type': leaf_node.node_type,
                      'has_tables': len(leaf_node.tables) > 0,
                      'table_count': len(leaf_node.tables),
                      'is_root_content': leaf_node_index == 0
                  }
              )
              chunks.append(chunk)

      return chunks





    def create_all_chunks(self, max_chunk_size: int = 4096,
                        min_overlap: int = 100,
                        max_overlap: int = 400) -> List[TextChunk]:
        """為所有葉子節點創建文本分塊"""
        all_chunks = []
        leaf_nodes = self.get_leaf_nodes()

        # log(f"Found {len(leaf_nodes)} leaf nodes")

        # 如果沒有葉子節點，檢查所有可能包含內容的節點
        if not leaf_nodes:
            # log("No leaf nodes found, checking all nodes for content...")

            content_nodes = []
            for i, node in enumerate(self.nodes):
                if node.text and node.text.strip():
                    content_nodes.append(i)
            #         log(f"  Node {i}: {len(node.text)} chars - {node.text[:50]}...")

            # log(f"Found {len(content_nodes)} content nodes")

            # 特別處理根節點（索引0）
            if 0 in content_nodes and len(content_nodes) == 1:
                # log("Only root node has content, treating as single content document")
                chunks = self.create_chunks_for_leaf_node(
                    0, max_chunk_size, min_overlap, max_overlap
                )
                all_chunks.extend(chunks)
                # log(f"  Created {len(chunks)} chunks from root node")
            else:
                # 處理其他內容節點
                for node_idx in content_nodes:
                    if node_idx == 0:  # 跳過根節點，除非它是唯一的內容節點
                        continue
                    # log(f"Processing content node {node_idx}: {self.nodes[node_idx].text[:50]}...")
                    chunks = self.create_chunks_for_leaf_node(
                        node_idx, max_chunk_size, min_overlap, max_overlap
                    )
                    all_chunks.extend(chunks)
                    # log(f"  Created {len(chunks)} chunks")

        else:
            # 正常處理葉子節點
            for leaf_idx in leaf_nodes:
                # log(f"Processing leaf node {leaf_idx}: {self.nodes[leaf_idx].text[:50]}...")
                chunks = self.create_chunks_for_leaf_node(
                    leaf_idx, max_chunk_size, min_overlap, max_overlap
                )
                all_chunks.extend(chunks)
                log(f"  Created {len(chunks)} chunks")

        return all_chunks






    def get_tree_structure(self, node_index: int = 0, indent: int = 0) -> str:
        """獲取樹結構的字符串表示"""
        if node_index >= len(self.nodes):
            return ""

        node = self.nodes[node_index]

        if node.node_type == "root":
            result = f"ROOT (children: {len(node.children_indices)})\n"
        else:
            result = "  " * indent + f"[{node.node_type}] Level {node.level}: {node.text[:50]}"
            if len(node.text) > 50:
                result += "..."
            result += f" (children: {len(node.children_indices)})\n"

        if node.tables:
            result += "  " * (indent + 1) + f"Tables: {len(node.tables)}\n"

        for child_index in node.children_indices:
            result += self.get_tree_structure(child_index, indent + 1)

        return result