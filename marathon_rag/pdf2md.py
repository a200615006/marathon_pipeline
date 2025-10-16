import os
import shutil
import sys
from pathlib import Path
from typing import List
# from data_processor.processing import process_documents_with_advanced_options
import logging

# 基本配置
logging.basicConfig(level=logging.INFO)
log = logging.info  # 或者使用 logger
# 添加项目根目录到系统路径，以便可以导入模块
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_processor.pdf2md import pdf_to_markdown  
from src.config import DOCUMENTS_DIR

if __name__ == "__main__":
    # --- 設定區 ---
    DATA_FOLDER = DOCUMENTS_DIR # 原始数据存放文件夹，格式可以包括 pdf,docx,md,xlsx,xls

    enable_docling = True  # 是否启用 docling 解析 PDF -> MD
    docling_source_folder = './pdf_source_for_docling'  # 存放被迁移的原始 PDF 文件的文件夹
    docling_output_suffix = '_pdf_parsed'  # 若重名时，MD 文件后缀

    # 若启用 docling，确保 PDF 临时存放文件夹存在
    if enable_docling and not os.path.exists(docling_source_folder):
        os.makedirs(docling_source_folder)
        log(f"Docling PDF 暫存資料夾 '{docling_source_folder}' 已建立。")

    def list_target_files(folder: str) -> List[str]:
        try:
            file_list = os.listdir(folder)
            # 保持原有支持的後綴
            return [f for f in file_list if f.lower().endswith((
                '.pdf', '.docx', '.md', '.xlsx', '.xls', 'txt', '.doc'
            ))]
        except FileNotFoundError:
            return []

    # 先掃描
    log(DATA_FOLDER)
    target_files = list_target_files(DATA_FOLDER)

    if enable_docling:
        # 1) 遷移 PDF 到 docling_source_folder
        pdfs_in_data = [f for f in target_files if f.lower().endswith('.pdf')]
        if pdfs_in_data:
            log(f"啟用 Docling，將 PDF 遷移至 '{docling_source_folder}' 進行解析: {pdfs_in_data}")
            for pdf_name in pdfs_in_data:
                src_path = os.path.join(DATA_FOLDER, pdf_name)
                dst_path = os.path.join(docling_source_folder, pdf_name)
                # 若目標已存在，先刪除或重命名，這裡選擇覆蓋
                if os.path.exists(dst_path):
                    os.remove(dst_path)
                shutil.move(src_path, dst_path)

            # 2) 使用 docling 將 PDF 轉為 MD，輸出回 DATA_FOLDER
            success_count = 0
            failed_count = 0
            failed_files = []
            
            for pdf_name in pdfs_in_data:
                pdf_src = os.path.join(docling_source_folder, pdf_name)
                base_no_ext = os.path.splitext(pdf_name)[0]
                md_candidate = f"{base_no_ext}.md"
                md_output_path = os.path.join(DATA_FOLDER, md_candidate)

                # 如果 DATA_FOLDER 已有同名 MD，則加上後綴
                if os.path.exists(md_output_path):
                    md_output_path = os.path.join(
                        DATA_FOLDER, f"{base_no_ext}{docling_output_suffix}.md"
                    )

                try:
                    log(f"使用 docling 解析: {pdf_src} -> {md_output_path}")
                    r=pdf_to_markdown(pdf_src, md_output_path, verbose=True)
                    if not r:
                        log(f"解析失敗: {pdf_src}, 錯誤: {e}")
                        # 将解析失败的PDF文件移回原来的存储目录
                        original_path = os.path.join(DATA_FOLDER, pdf_name)
                        try:
                            shutil.move(pdf_src, original_path)
                            log(f"已将解析失败的PDF文件移回原目录: {original_path}")
                        except Exception as move_error:
                            log(f"移回文件失败: {pdf_src} -> {original_path}, 错误: {move_error}")
                        
                        failed_count += 1
                        failed_files.append(pdf_src)
                    else:
                        success_count += 1
                except Exception as e:
                    log(f"解析失敗: {pdf_src}, 錯誤: {e}")
                    # 将解析失败的PDF文件移回原来的存储目录
                    original_path = os.path.join(DATA_FOLDER, pdf_name)
                    try:
                        shutil.move(pdf_src, original_path)
                        log(f"已将解析失败的PDF文件移回原目录: {original_path}")
                    except Exception as move_error:
                        log(f"移回文件失败: {pdf_src} -> {original_path}, 错误: {move_error}")
                    
                    failed_count += 1
                    failed_files.append(pdf_src)
            
            # 输出解析总结
            log("\n" + "="*50)
            log("PDF 解析总结")
            log("="*50)
            log(f"成功解析数量: {success_count}")
            log(f"失败解析数量: {failed_count}")
            log(f"总处理文件数量: {success_count + failed_count}")
            
            if failed_files:
                log("\n失败文件列表:")
                for i, file_path in enumerate(failed_files, 1):
                    log(f"{i}. {file_path}")
            
            log("="*50)
