#!/usr/bin/env python3
"""
简化的PDF到Markdown转换工具
提供单一函数接口：pdf_to_markdown(pdf_path, output_path)
"""

import tempfile
import shutil
import sys
from pathlib import Path
import logging

# 基本配置
logging.basicConfig(level=logging.INFO)
log = logging.info  # 或者使用 logger
# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_processor.src_docling_pdf.pdf_parsing import PDFParser
from src.data_processor.src_docling_pdf.parsed_reports_merging import PageTextPreparation
from src.config import PDF_MODEL_PATH

def pdf_to_markdown(pdf_path: str | Path, output_path: str | Path, verbose: bool = False) -> bool:
    """
    将PDF文件转换为Markdown格式
    
    Args:
        pdf_path: 输入PDF文件路径
        output_path: 输出Markdown文件路径
        verbose: 是否显示详细日志
        
    Returns:
        bool: 转换成功返回True，失败返回False
    """
    try:
        # 转换路径为Path对象
        pdf_path = Path(pdf_path)
        output_path = Path(output_path)
        
        # 检查输入文件是否存在
        if not pdf_path.exists():
            if verbose:
                log(f"错误: PDF文件不存在 - {pdf_path}")
            return False
        
        if not pdf_path.suffix.lower() == '.pdf':
            if verbose:
                log(f"错误: 输入文件不是PDF格式 - {pdf_path}")
            return False
        
        # 创建输出目录（如果不存在）
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建临时工作目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 设置临时目录结构
            parsed_reports_path = temp_path / "parsed_reports"
            parsed_reports_debug_path = temp_path / "parsed_reports_debug"
            markdown_output_path = temp_path / "markdown_output"
            
            # 创建必要的目录
            parsed_reports_path.mkdir(exist_ok=True)
            parsed_reports_debug_path.mkdir(exist_ok=True)
            markdown_output_path.mkdir(exist_ok=True)
            
            if verbose:
                log(f"开始解析PDF文件: {pdf_path.name}")
            
            # 步骤1: 初始化PDF解析器并解析
            pdf_parser = PDFParser(
                output_dir=parsed_reports_path,
                use_local_models=True,
                local_models_base_path=PDF_MODEL_PATH,
                num_threads=4,
                verbose=verbose
            )
            pdf_parser.debug_data_path = parsed_reports_debug_path
            
            # 解析PDF
            pdf_parser.parse_and_export(input_doc_paths=[pdf_path])
            if verbose:
                log("PDF解析完成")
            
            # 显式清理PDF解析器资源
            pdf_parser.clear_gpu_memory()
            
            # 步骤2: 导出为Markdown
            if verbose:
                log("开始导出为Markdown格式...")
            ptp = PageTextPreparation(use_serialized_tables=False)
            
            ptp.export_to_markdown(
                reports_dir=parsed_reports_path,
                output_dir=markdown_output_path
            )
            if verbose:
                log("Markdown导出完成")
            
            # 步骤3: 复制结果到目标位置
            # 查找生成的markdown文件（通常与PDF同名）
            pdf_stem = pdf_path.stem
            markdown_files = list(markdown_output_path.glob("*.md"))
            
            if not markdown_files:
                if verbose:
                    log("错误: 未生成Markdown文件")
                return False
            
            # 如果有多个markdown文件，优先选择与PDF同名的，否则选择第一个
            target_md_file = None
            for md_file in markdown_files:
                if md_file.stem == pdf_stem:
                    target_md_file = md_file
                    break
            
            if target_md_file is None:
                target_md_file = markdown_files[0]
            
            # 复制到目标路径
            shutil.copy2(target_md_file, output_path)
            if verbose:
                log(f"转换完成: {output_path}")
            
            return True
            
    except Exception as e:
        if verbose:
            log(f"转换过程中出现错误: {str(e)}")
        return False

def batch_pdf_to_markdown(input_dir: str | Path, output_dir: str | Path, verbose: bool = False) -> dict:
    """
    批量转换PDF文件为Markdown格式
    
    Args:
        input_dir: 包含PDF文件的输入目录
        output_dir: Markdown文件输出目录
        verbose: 是否显示详细日志
        
    Returns:
        dict: 转换结果统计 {'success': int, 'failed': int, 'failed_files': list}
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        if verbose:
            log(f"错误: 输入目录不存在 - {input_dir}")
        return {'success': 0, 'failed': 1, 'failed_files': [str(input_dir)]}
    
    # 获取所有PDF文件
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        if verbose:
            log(f"在目录 {input_dir} 中未找到PDF文件")
        return {'success': 0, 'failed': 0, 'failed_files': []}
    
    if verbose:
        log(f"找到 {len(pdf_files)} 个PDF文件待处理")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    failed_count = 0
    failed_files = []
    
    for pdf_file in pdf_files:
        output_file = output_dir / f"{pdf_file.stem}.md"
        if verbose:
            log(f"\n处理文件: {pdf_file.name}")
        
        if pdf_to_markdown(pdf_file, output_file, verbose=verbose):
            success_count += 1
        else:
            failed_count += 1
            failed_files.append(str(pdf_file))
    
    if verbose:
        log(f"\n批量转换完成:")
        log(f"成功: {success_count} 个文件")
        log(f"失败: {failed_count} 个文件")
        
        if failed_files:
            log("失败的文件:")
            for file in failed_files:
                log(f"  - {file}")
    
    return {
        'success': success_count,
        'failed': failed_count,
        'failed_files': failed_files
    }

# 使用示例
if __name__ == "__main__":
    # 单文件转换示例
    # pdf_to_markdown("/root/rag/data-pdftest/腾讯财报 2025 简体.pdf", "./output.md")
    
    # 批量转换示例
    batch_pdf_to_markdown("./pdf", "./extracted_md2")
    
    log("PDF到Markdown转换工具已就绪")
    log("使用方法:")
    log("1. 单文件转换: pdf_to_markdown('input.pdf', 'output.md')")
    log("2. 批量转换: batch_pdf_to_markdown('input_dir', 'output_dir')")
