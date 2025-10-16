import os
import re
import traceback
from typing import List

# 從我們自己的模組中導入功能
from .parsers import (
    extract_outline_from_docx_with_tables,
    # extract_outline_from_doc_with_tables,
    extract_outline_from_pdf,
    extract_outline_from_markdown,
    extract_outline_from_excel
)
from .document_tree import DocumentTree
from .chunking import merge_chunks_by_path
from .utils import (
    save_chunks_to_files,
    analyze_document_structure,
    export_chunks_to_json,
    create_chunk_index,
    is_table_header # parse_extracted_text 會用到
)

import logging

# 基本配置
logging.basicConfig(level=logging.INFO)
log = logging.info  # 或者使用 logger
def parse_extracted_text(text_content: str) -> DocumentTree:
    """解析提取的文本內容，構建文檔樹"""
    tree = DocumentTree()
    lines = text_content.split('\n')


    # 檢查是否有任何標題結構
    has_headings = any(line.strip().startswith('Heading') for line in lines)


    # 如果完全沒有標題結構，將所有內容作為單一段落處理
    if not has_headings:
        log("No headings found, treating as single paragraph document")
        all_text = '\n'.join([line for line in lines if line.strip() and not line.startswith('Outline')])

        # 直接添加到根節點
        if tree.nodes[0].text:
            tree.nodes[0].text += "\n" + all_text
        else:
            tree.nodes[0].text = all_text

        return tree


    current_tables = []  # 暫存表格數據
    cache_lines=''

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # log(f"Processing line: {line}")

        # 解析標題
        heading_match = re.match(r'^(\s*)Heading (\d+): (.+)$', line)
        if heading_match:
            if cache_lines:
                # log(f"adding paragraph: {content}")
                tree.add_paragraph(cache_lines, 999, "paragraph")
                cache_lines=''

            indent, level_str, title = heading_match.groups()
            level = int(level_str)

            # 如果有暫存的表格，添加到上一個節點
            if current_tables:
                for table in current_tables:
                    tree.add_table_to_current_node(table)
                current_tables = []

            tree.add_paragraph(title, level, "heading")
            continue

        # 解析段落
        paragraph_match = re.match(r'^Paragraph: (.+)$', line)
        if paragraph_match:
            if cache_lines:
                # log(f"adding paragraph: {content}")
                tree.add_paragraph(cache_lines, 999, "paragraph")
                cache_lines=''

            content = paragraph_match.group(1)
            # log(f"Processing paragraph: {content}")

            # 檢查是否是表格行
            if '|' in content and content.count('|') >= 2:
                # 這是表格數據，暫存起來
                table_row = [cell.strip() for cell in content.split('|')[1:-1]]

                # 如果是新表格的開始
                if not current_tables:
                    current_tables.append({
                        'rows': [table_row],
                        'has_header': is_table_header(table_row),
                        'position': 'after_current_heading'
                    })
                else:
                    # 添加到當前表格
                    current_tables[-1]['rows'].append(table_row)
            else:
                # 如果有暫存的表格，先添加到當前節點
                if current_tables:
                    for table in current_tables:
                        tree.add_table_to_current_node(table)
                    current_tables = []

                # 正文段落
                cache_lines += content + '\n'
            continue
        else:
            if current_tables:
                # 當前行不是表格行，將暫存的表格添加到上一個節點
                for table in current_tables:
                    tree.add_table_to_current_node(table)
                current_tables = []
            else:
                cache_lines += line + '\n'


        # 解析表格標題
        table_match = re.match(r'^Table \d+ \(Page \d+\):$', line)
        if table_match:
            # 表格開始標記，準備收集表格數據
            continue
        
        

    # 處理剩餘的表格
    if current_tables:
        for table in current_tables:
            tree.add_table_to_current_node(table)

    return tree

    
def process_documents_with_advanced_options(target_files, data_folder,                                                                                      
                                           chunks_output_folder=None,
                                           documents_output_folder=None,
                                           activate_docling_pdf=False,
                                           save_full_text=True,
                                           save_tree_structure=True,
                                           save_chunk_summary=True,
                                           filename_prefix="",
                                           # 现有选项
                                           enable_chunking=True,
                                           max_chunk_size=4096,
                                           min_overlap=100,
                                           max_overlap=400,
                                           enable_path_merging=False,
                                           export_json=True,
                                           create_index=True,
                                           analyze_structure=True,
                                           # 新增增量处理选项
                                           incremental_processing=False):
    """增强版文档处理函数，包含增量处理选项"""

    # 设置输出文件夹，如果未指定则使用默认
    if documents_output_folder is None:
        documents_output_folder = output_folder
    if chunks_output_folder is None:
        chunks_output_folder = output_folder
    
    os.makedirs(documents_output_folder, exist_ok=True)
    os.makedirs(chunks_output_folder, exist_ok=True)

    # 增量处理：过滤已处理的文件
    if incremental_processing:
        processed_files = set()
        
        # 检查文档输出文件夹中的文件
        if os.path.exists(documents_output_folder):
            for filename in os.listdir(documents_output_folder):
                if filename.endswith('.txt') or filename.endswith('_tree.txt'):
                    base_name = filename.replace('.txt', '').replace('_tree', '')
                    processed_files.add(base_name)
        
        # 检查分块输出文件夹中的文件
        if os.path.exists(chunks_output_folder):
            for filename in os.listdir(chunks_output_folder):
                if filename.endswith('_chunks.json') or filename.endswith('_chunk_index.json'):
                    base_name = filename.replace('_chunks.json', '').replace('_chunk_index.json', '')
                    processed_files.add(base_name)
        
        # 过滤掉已处理的文件
        original_count = len(target_files)
        target_files = [
            file_name for file_name in target_files 
            if file_name.rsplit('.', 1)[0] not in processed_files
        ]
        log(f"增量处理模式: 从 {original_count} 个文件中过滤出 {len(target_files)} 个新文件")

    for file_name in target_files:
        file_path = os.path.join(data_folder, file_name)

        if not os.path.exists(file_path):
            log(f"文件未找到: {file_path}")
            continue

        try:
            log(f"处理中: {file_name}")

            if file_name.lower().endswith('.docx'):
                extracted_content = extract_outline_from_docx_with_tables(file_path)
            # elif file_name.lower().endswith('.doc'):
            #     extracted_content = extract_outline_from_doc_with_tables(file_path)
            elif file_name.lower().endswith('.pdf'):
                
                
                extracted_content = extract_outline_from_pdf(file_path)
            elif file_name.lower().endswith('.md'):
                extracted_content = extract_outline_from_markdown(file_path)
            elif file_name.lower().endswith(('.xlsx', '.xls')):
                extracted_content = extract_outline_from_excel(file_path)
            elif file_name.lower().endswith('.txt'):
                # 读取txt文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    extracted_content = f.read()
                # 为txt文件创建简单的文档结构
                extracted_content = f"# {file_name}\n\n{extracted_content}"
            else:
                log(f"不支持的文件类型: {file_name}")
                continue

            doc_tree = parse_extracted_text(extracted_content)
            base_filename = file_name.rsplit('.', 1)[0]

            if analyze_structure:
                analyze_document_structure(doc_tree)

            # 根据选项保存基础文件到文档输出文件夹
            if save_full_text:
                output_filename = base_filename + '.txt'
                output_path = os.path.join(documents_output_folder, output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_content)
                log(f"已保存全文到 {output_path}")

            if save_tree_structure:
                tree_filename = base_filename + '_tree.txt'
                tree_path = os.path.join(documents_output_folder, tree_filename)
                with open(tree_path, 'w', encoding='utf-8') as f:
                    f.write("文档树结构:\n")
                    f.write("=" * 50 + "\n")
                    f.write(doc_tree.get_tree_structure())
                log(f"已保存树结构到 {tree_path}")

            if enable_chunking:
                # log("创建分块中...")
                chunks = doc_tree.create_all_chunks(
                    max_chunk_size=max_chunk_size,
                    min_overlap=min_overlap,
                    max_overlap=max_overlap
                )

                if enable_path_merging and chunks:
                    # log("应用基于路径的合并...")
                    original_count = len(chunks)
                    chunks = merge_chunks_by_path(chunks, max_chunk_size, max_overlap)
                    log(f"路径合并将分块从 {original_count} 减少到 {len(chunks)}")

                if chunks:
                    # 传入前缀以保存分块文件到chunk输出文件夹
                    chunks_folder = save_chunks_to_files(chunks, chunks_output_folder, base_filename, filename_prefix)

                    # 选择性保存 chunk summary
                    if not save_chunk_summary:
                        summary_path = os.path.join(chunks_folder, "chunks_summary.txt")
                        if os.path.exists(summary_path):
                            os.remove(summary_path)
                            # log("根据设置已移除分块摘要文件。")

                    if export_json:
                        json_path = os.path.join(chunks_output_folder, f"{base_filename}_chunks.json")
                        export_chunks_to_json(chunks, json_path)

                    if create_index:
                        chunk_index = create_chunk_index(chunks)
                        index_path = os.path.join(chunks_output_folder, f"{base_filename}_chunk_index.json")
                        import json
                        with open(index_path, 'w', encoding='utf-8') as f:
                            json.dump(chunk_index, f, ensure_ascii=False, indent=2)
                        # log(f"已创建分块索引: {index_path}")

                    # log(f"成功处理 {file_name}，生成 {len(chunks)} 个分块")
                else:
                    log("未创建任何分块")


        except Exception as e:
            log(f"处理 {file_name} 时出错: {e}")
            import traceback
            traceback.print_exc()

