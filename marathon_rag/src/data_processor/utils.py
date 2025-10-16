import os
import json
from typing import List, Dict, Any
# 從 models 模組導入數據類，避免循環引用
from .models import TextChunk
from .document_tree import DocumentTree
import re
import logging

# 基本配置
logging.basicConfig(level=logging.INFO)
log = logging.info  # 或者使用 logger
def is_likely_heading(text):
    """
    簡單的啟發式判斷文本是否可能是標題
    """
    # 標題通常較短、首字母大寫、可能包含數字編號
    if len(text) > 100:  # 太長不太可能是標題
        return False

    # 檢查是否有編號格式（如 1.1, 第一章 等）
    heading_patterns = [
        r'^\d+\.?\d*\.?\d*\s+',  # 1. 1.1 1.1.1 格式
        r'^第[一二三四五六七八九十\d]+[章節条款部分]\s*',  # 第一章、第二節等
        r'^[一二三四五六七八九十]+[、\.]\s*',  # 一、二、三、格式
        r'^[A-Z][A-Z\s]*$',  # 全大寫
        r'^[A-Z][a-z\s]+$'   # 首字母大寫的短句
    ]

    for pattern in heading_patterns:
        if re.match(pattern, text):
            return True

    # 檢查是否全部大寫或首字母大寫且較短
    if text.isupper() or (text[0].isupper() and len(text) < 50):
        return True

    return False

def infer_heading_level(text):
    """
    推斷標題級別
    """
    # 根據編號格式推斷級別
    if re.match(r'^\d+\.\d+\.\d+', text):  # 1.1.1 格式
        return 3
    elif re.match(r'^\d+\.\d+', text):     # 1.1 格式
        return 2
    elif re.match(r'^\d+\.?', text):       # 1. 格式
        return 1
    elif re.match(r'^第[一二三四五六七八九十\d]+章', text):  # 第一章
        return 1
    elif re.match(r'^第[一二三四五六七八九十\d]+節', text):  # 第一節
        return 2
    elif re.match(r'^[一二三四五六七八九十]+[、\.]', text):  # 一、
        return 2
    elif text.isupper():  # 全大寫可能是高級標題
        return 1
    else:
        return 2  # 默認為二級標題



def is_table_header(row: List[str]) -> bool:
    """簡單判斷表格行是否為表頭（基於文本特徵）"""
    if not row:
        return False

    # 檢查是否包含常見的表頭關鍵詞
    header_keywords = [
        # 基本信息類
        '名称', '姓名', '编号', '序号', '代码', '标识', '标号', 'ID', 'Code', 'Name',

        # 分類描述類
        '类型', '分类', '种类', '类别', '性质', '属性', '特征', 'Type', 'Category',
        '项目', '内容', '描述', '说明', '备注', '注释', '详情', 'Item', 'Description',

        # 數量金額類
        '数量', '金额', '价格', '费用', '成本', '总计', '小计', '合计',
        'Amount', 'Price', 'Cost', 'Total', 'Sum', 'Count', 'Quantity',

        # 時間日期類
        '日期', '时间', '年份', '月份', '期间', '开始', '结束', '截止',
        'Date', 'Time', 'Year', 'Month', 'Period', 'Start', 'End',

        # 狀態結果類
        '状态', '结果', '等级', '级别', '评分', '得分', '排名', '排序',
        'Status', 'Result', 'Level', 'Grade', 'Score', 'Rank',

        # 地址聯系類
        '地址', '位置', '区域', '部门', '单位', '公司', '机构', '组织',
        'Address', 'Location', 'Department', 'Company', 'Organization',
        '电话', '邮箱', '联系', '手机', 'Phone', 'Email', 'Contact',

        # 人員相關類
        '负责人', '联系人', '经办人', '申请人', '审批人', '操作员',
        'Manager', 'Contact', 'Operator', 'Applicant', 'Approver',

        # 業務流程類
        '流程', '步骤', '阶段', '环节', '操作', '处理', '审核', '审批',
        'Process', 'Step', 'Stage', 'Operation', 'Review', 'Approval',

        # 技術參數類
        '参数', '配置', '规格', '型号', '版本', '尺寸', '重量', '容量',
        'Parameter', 'Config', 'Specification', 'Model', 'Version', 'Size',

        # 統計分析類
        '比例', '百分比', '占比', '增长率', '完成率', '达成率', '通过率','准确率'
        'Ratio', 'Percentage', 'Rate', 'Growth', 'Completion','accuracy'

        # 質量安全類
        '质量', '安全', '风险', '问题', '缺陷', '异常', '故障',
        'Quality', 'Safety', 'Risk', 'Issue', 'Defect', 'Exception',

        # 計劃目標類
        '计划', '目标', '任务', '指标', '要求', '标准', '规范',
        'Plan', 'Target', 'Task', 'Indicator', 'Requirement', 'Standard',

        # 資源材料類
        '资源', '材料', '设备', '工具', '软件', '系统', '平台',
        'Resource', 'Material', 'Equipment', 'Tool', 'Software', 'System',

        # 權限角色類
        '权限', '角色', '职位', '岗位', '职责', '权利', '义务',
        'Permission', 'Role', 'Position', 'Responsibility', 'Authority'
    ]

    header_score = 0
    total_cells = len([cell for cell in row if cell and cell.strip()])

    if total_cells == 0:
        return False

    for cell in row:
        if cell and cell.strip():
            cell_text = cell.strip()

            # 直接匹配關鍵詞
            for keyword in header_keywords:
                if keyword in cell_text:
                    header_score += 1
                    break
            else:
                # 檢查是否是簡短的描述性文字（可能是表頭）
                if len(cell_text) <= 10 and not cell_text.isdigit():
                    # 檢查是否包含中文字符或英文字母
                    if any('\u4e00' <= char <= '\u9fff' for char in cell_text) or \
                       any(char.isalpha() for char in cell_text):
                        header_score += 0.5

    # 如果超過60%的單元格符合表頭特徵，認為是表頭
    return header_score / total_cells > 0.6


def save_chunks_to_files(chunks: List[TextChunk], output_folder: str, base_filename: str, filename_prefix: str = ""):
    """将分块保存到文件（极简命名：顺序号）"""
    import os

    chunks_folder = os.path.join(output_folder, f"{base_filename}_chunks")
    os.makedirs(chunks_folder, exist_ok=True)

    # 尝试使用你在 chunking.py 中的去路径标题函数
    def _get_body_text(chunk: TextChunk) -> str:
        titles = getattr(chunk, 'path_titles', []) or []
        content = getattr(chunk, 'content', '') or ''
        try:
            # 若 remove_path_title_from_content 在同模块可直接调用，如不在同包可换成直接返回 content
            from .chunking import remove_path_title_from_content
            return remove_path_title_from_content(content, titles).strip()
        except Exception:
            return content

    # 固定宽度序号，如 0001、0002
    width = max(4, len(str(len(chunks))))
    for idx, chunk in enumerate(chunks, start=1):
        seq = str(idx).zfill(width)
        # 文件名只由 base + 序号 组成，可选加 prefix
        chunk_filename = f"{filename_prefix}{base_filename}_{seq}.txt"
        chunk_path = os.path.join(chunks_folder, chunk_filename)
        # log(base_filename)

        body = _get_body_text(chunk)
        with open(chunk_path, 'w', encoding='utf-8') as f:
            f.write(f"Chunk path: {chunk_path}\n")
            f.write(f"Path Titles: {' > '.join(chunk.path_titles)}\n")
            f.write(body if body else (chunk.content or ""))

    log(f"Saved {len(chunks)} chunks to {chunks_folder}")
    return chunks_folder





def analyze_document_structure(doc_tree: DocumentTree):
    """分析文檔結構的輔助函數"""
    log("Document Structure Analysis:")
    log("=" * 50)

    total_nodes = len(doc_tree.nodes)
    heading_nodes = sum(1 for node in doc_tree.nodes if node.node_type == "heading")
    paragraph_nodes = sum(1 for node in doc_tree.nodes if node.node_type == "paragraph")
    leaf_nodes = len(doc_tree.get_leaf_nodes())

    log(f"Total nodes: {total_nodes}")
    log(f"Heading nodes: {heading_nodes}")
    log(f"Paragraph nodes: {paragraph_nodes}")
    log(f"Leaf nodes: {leaf_nodes}")
    log()

    # 分析層級分佈
    level_distribution = {}
    for node in doc_tree.nodes:
        if node.node_type == "heading":
            level = node.level
            if level not in level_distribution:
                level_distribution[level] = 0
            level_distribution[level] += 1

    log("Heading level distribution:")
    for level in sorted(level_distribution.keys()):
        log(f"  Level {level}: {level_distribution[level]} headings")
    log()

    # 分析文本長度
    text_lengths = [len(node.text) for node in doc_tree.nodes if node.text.strip()]
    if text_lengths:
        avg_length = sum(text_lengths) / len(text_lengths)
        max_length = max(text_lengths)
        min_length = min(text_lengths)

        log(f"Text length statistics:")
        log(f"  Average: {avg_length:.1f} characters")
        log(f"  Maximum: {max_length} characters")
        log(f"  Minimum: {min_length} characters")
    log()

def export_chunks_to_json(chunks: List[TextChunk], output_path: str):
    """將分塊導出為JSON格式"""
    import json

    chunks_data = []
    for chunk in chunks:
        chunk_data = {
            'chunk_id': chunk.chunk_id,
            'content': chunk.content,
            'path_titles': chunk.path_titles,
            'source_node_index': chunk.source_node_index,
            'chunk_index': chunk.chunk_index,
            'metadata': chunk.metadata
        }
        chunks_data.append(chunk_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    log(f"Exported {len(chunks)} chunks to JSON: {output_path}")

def create_chunk_index(chunks: List[TextChunk]) -> Dict[str, Any]:
    """創建分塊索引，便於查找和管理"""
    index = {
        'total_chunks': len(chunks),
        'chunks_by_node': {},
        'chunks_by_path': {},
        'chunk_metadata': {}
    }

    for chunk in chunks:
        # 按源節點索引
        node_idx = chunk.source_node_index
        if node_idx not in index['chunks_by_node']:
            index['chunks_by_node'][node_idx] = []
        index['chunks_by_node'][node_idx].append(chunk.chunk_id)

        # 按路徑索引
        path_key = ' > '.join(chunk.path_titles)
        if path_key not in index['chunks_by_path']:
            index['chunks_by_path'][path_key] = []
        index['chunks_by_path'][path_key].append(chunk.chunk_id)

        # 元數據索引
        index['chunk_metadata'][chunk.chunk_id] = {
            'content_length': len(chunk.content),
            'path_titles': chunk.path_titles,
            'source_node': chunk.source_node_index,
            'chunk_index': chunk.chunk_index,
            'metadata': chunk.metadata
        }

    return index
