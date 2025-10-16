import os
import time
import logging
import re
import json
from tabulate import tabulate
from pathlib import Path
from typing import Iterable, List
import torch

# from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
# from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult

_log = logging.getLogger(__name__)
import logging

# 基本配置
logging.basicConfig(level=logging.INFO)
log = logging.info  # 或者使用 logger

def _process_chunk(pdf_paths, pdf_backend, output_dir, num_threads, metadata_lookup, debug_data_path):
    """Helper function to process a chunk of PDFs in a separate process."""
    # Create a new parser instance for this process
    parser = PDFParser(
        pdf_backend=pdf_backend,
        output_dir=output_dir,
        num_threads=num_threads,
        csv_metadata_path=None  # Metadata lookup is passed directly
    )
    parser.metadata_lookup = metadata_lookup
    parser.debug_data_path = debug_data_path
    parser.parse_and_export(pdf_paths)
    return f"Processed {len(pdf_paths)} PDFs."


class PDFParser:
    def __init__(
            self,
            pdf_backend=DoclingParseV2DocumentBackend,
            output_dir: Path = Path("./parsed_pdfs"),
            num_threads: int = None,
            csv_metadata_path: Path = None,
            artifacts_path: str = None,
            use_local_models: bool = True,  # 新增参数
            local_models_base_path: str = "./model",  # 新增：本地模型基础路径,
            enable_gpu: bool = True,  # 新增GPU启用选项
            verbose: bool = False,  # 新增：控制日志输出
    ):
        self.pdf_backend = pdf_backend
        self.output_dir = output_dir
        self.num_threads = num_threads
        self.metadata_lookup = {}
        self.debug_data_path = None
        self.artifacts_path = artifacts_path
        self.use_local_models = use_local_models
        self.local_models_base_path = Path(local_models_base_path)  # 保存基础路径
        self.enable_gpu = enable_gpu
        self.verbose = verbose  # 保存verbose参数

        # 检查GPU可用性
        if self.enable_gpu:
            self.gpu_available = self.check_gpu_availability()
        else:
            self.gpu_available = False
            log("✓ GPU支持已禁用，使用CPU处理")

        # 设置本地模型环境变量
        if self.use_local_models:
            self._setup_local_models()

        # 设置环境变量（如果提供了路径）
        if self.artifacts_path:
            os.environ["DOCLING_SERVE_ARTIFACTS_PATH"] = self.artifacts_path

        self.doc_converter = self._create_document_converter()

        if csv_metadata_path is not None:
            self.metadata_lookup = self._parse_csv_metadata(csv_metadata_path)

        if self.num_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(self.num_threads)

    def check_gpu_availability(self):
        """检查GPU可用性"""
        import torch

        if self.verbose:
            log("=== GPU 可用性检查 ===")

        # 检查CUDA
        if torch.cuda.is_available():
            if self.verbose:
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024 ** 3

                log(f"✓ CUDA 可用")
                log(f"  GPU 数量: {gpu_count}")
                log(f"  当前设备: {current_device}")
                log(f"  GPU 名称: {gpu_name}")
                log(f"  GPU 内存: {gpu_memory:.1f} GB")
            return True

        # 检查MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if self.verbose:
                log(f"✓ MPS (Apple Silicon GPU) 可用")
            return True

        else:
            if self.verbose:
                log(f"✗ 未检测到可用的GPU，将使用CPU")
            return False

    def _setup_local_models(self):
        """设置本地模型路径的环境变量"""
        import os
        from pathlib import Path

        # 获取绝对路径
        base_path = self.local_models_base_path

        if self.verbose:
            log(f"=== 本地模型配置 ===")
            log(f"基础路径: {base_path}")

        # 设置 HuggingFace 缓存目录为你的 model 目录
        # 这样 HF 会直接从你的目录结构中读取模型
        os.environ["HF_HUB_CACHE"] = str(base_path)
        os.environ["HF_HOME"] = str(base_path)
        os.environ["HF_DATASETS_CACHE"] = str(base_path)
        if self.verbose:
            log(f"✓ 设置 HuggingFace 缓存目录: {base_path}")

        # 设置 EasyOCR 模型目录
        easyocr_dir = base_path / "EasyOCR"
        if easyocr_dir.exists():
            os.environ["EASYOCR_MODULE_PATH"] = str(easyocr_dir)
            if self.verbose:
                log(f"✓ 设置 EasyOCR 模型目录: {easyocr_dir}")
        elif self.verbose:
            log(f"✗ EasyOCR 目录不存在: {easyocr_dir}")

        # 强制离线模式，防止尝试下载
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        if self.verbose:
            log(f"✓ 启用离线模式")

        # 验证 Docling 模型路径
        docling_models_path = base_path / "models--ds4sd--docling-models"
        if docling_models_path.exists():
            if self.verbose:
                log(f"✓ 找到 Docling 模型目录: {docling_models_path}")

                # 检查快照目录
                snapshots_dir = docling_models_path / "snapshots"
                if snapshots_dir.exists():
                    log(f"  ✓ 快照目录存在: {snapshots_dir}")
                else:
                    log(f"  ✗ 快照目录不存在: {snapshots_dir}")
        elif self.verbose:
            log(f"✗ Docling 模型目录不存在: {docling_models_path}")

        if self.verbose:
            log("✓ 本地模型环境配置完成")

    @staticmethod
    def _parse_csv_metadata(csv_path: Path) -> dict:
        """Parse CSV file and create a lookup dictionary with sha1 as key."""
        import csv
        metadata_lookup = {}

        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Handle both old and new CSV formats for company name
                company_name = row.get('company_name', row.get('name', '')).strip('"')
                metadata_lookup[row['sha1']] = {
                    'company_name': company_name
                }
        return metadata_lookup

    def _create_document_converter(self) -> "DocumentConverter":
        """Creates and returns a DocumentConverter with default pipeline options."""
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions, \
            AcceleratorOptions, AcceleratorDevice
        from docling.datamodel.base_models import InputFormat
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
        import torch

        pipeline_options = PdfPipelineOptions()

        # GPU 配置保持不变...
        if torch.cuda.is_available():
            accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CUDA,
                num_threads=self.num_threads or 4
            )
            if self.verbose:
                log(f"✓ 启用CUDA GPU加速")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.MPS,
                num_threads=self.num_threads or 4
            )
            if self.verbose:
                log(f"✓ 启用MPS GPU加速 (Apple Silicon)")
        else:
            accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CPU,
                num_threads=self.num_threads or 4
            )
            if self.verbose:
                log(f"✓ 使用CPU处理")

        pipeline_options.accelerator_options = accelerator_options

        # 修改本地模型配置部分
        if self.artifacts_path:
            pipeline_options.artifacts_path = self.artifacts_path
            log(f"✓ 使用指定的 artifacts 路径: {self.artifacts_path}")
        elif self.use_local_models:
            base_path = self.local_models_base_path.resolve()
            
            # 修改：指向 model 目录而不是 model_artifacts 目录
            # 因为 Docling 会自动在后面添加 model_artifacts
            artifacts_path = base_path / "models--ds4sd--docling-models" / "snapshots" / "model"
            
            if artifacts_path.exists():
                pipeline_options.artifacts_path = str(artifacts_path)
                log(f"✓ 使用本地 artifacts 路径: {artifacts_path}")
                
                # 验证 model_artifacts 子目录存在
                model_artifacts_dir = artifacts_path / "model_artifacts"
                if model_artifacts_dir.exists():
                    log(f"  ✓ model_artifacts 目录存在")
                    
                    # 验证关键模型文件
                    layout_model = model_artifacts_dir / "layout" / "model.safetensors"
                    tableformer_accurate = model_artifacts_dir / "tableformer" / "accurate" / "tableformer_accurate.safetensors"
                    
                    if layout_model.exists():
                        log(f"  ✓ Layout 模型文件存在: {layout_model}")
                    else:
                        log(f"  ✗ Layout 模型文件不存在: {layout_model}")
                        
                    if tableformer_accurate.exists():
                        log(f"  ✓ TableFormer 模型文件存在: {tableformer_accurate}")
                    else:
                        log(f"  ✗ TableFormer 模型文件不存在: {tableformer_accurate}")
                else:
                    log(f"  ✗ model_artifacts 目录不存在: {model_artifacts_dir}")
            else:
                log(f"✗ artifacts 路径不存在: {artifacts_path}")

        pipeline_options.do_ocr = True

        # EasyOCR 配置
        if self.use_local_models:
            base_path = self.local_models_base_path.resolve()
            easyocr_model_dir = base_path / "EasyOCR" / "model"

            if easyocr_model_dir.exists():
                ocr_options = EasyOcrOptions(
                    lang=['en', 'ch_sim'],
                    force_full_page_ocr=False,
                    model_storage_directory=str(easyocr_model_dir)
                )
                log(f"✓ 使用本地 EasyOCR 模型目录: {easyocr_model_dir}")
            else:
                log(f"✗ EasyOCR 目录不存在，使用默认配置")
                ocr_options = EasyOcrOptions(lang=['en', 'ch_sim'], force_full_page_ocr=False)
        else:
            ocr_options = EasyOcrOptions(lang=['en', 'ch_sim'], force_full_page_ocr=False)

        pipeline_options.ocr_options = ocr_options
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
                backend=self.pdf_backend
            )
        }

        return DocumentConverter(format_options=format_options)

    def verify_local_models(self):
        """验证本地模型是否存在"""
        from pathlib import Path

        log("正在验证本地模型...")
        base_path = self.local_models_base_path.resolve()

        # 检查 Docling 模型
        docling_model_path = base_path / "docling_models_package" / "models--ds4sd--docling-models"
        if docling_model_path.exists():
            log(f"✓ 找到 Docling 模型目录: {docling_model_path}")

            # 检查快照目录
            snapshots_dir = docling_model_path / "snapshots"
            if snapshots_dir.exists():
                snapshot_dirs = list(snapshots_dir.glob("*"))
                if snapshot_dirs:
                    latest_snapshot = max(snapshot_dirs, key=lambda x: x.stat().st_mtime)
                    log(f"  ✓ 快照目录: {latest_snapshot}")

                    artifacts_path = latest_snapshot / "model" / "model_artifacts"
                    if artifacts_path.exists():
                        log(f"  ✓ Artifacts 路径: {artifacts_path}")

                        # 检查具体的模型文件
                        layout_path = artifacts_path / "layout"
                        tableformer_path = artifacts_path / "tableformer"

                        if layout_path.exists():
                            log(f"    ✓ Layout 模型: {layout_path}")
                        else:
                            log(f"    ✗ Layout 模型未找到: {layout_path}")

                        if tableformer_path.exists():
                            log(f"    ✓ TableFormer 模型: {tableformer_path}")
                        else:
                            log(f"    ✗ TableFormer 模型未找到: {tableformer_path}")
                    else:
                        log(f"  ✗ Artifacts 路径未找到: {artifacts_path}")
                else:
                    log(f"  ✗ 快照目录为空: {snapshots_dir}")
            else:
                log(f"  ✗ 快照目录未找到: {snapshots_dir}")
        else:
            log(f"✗ Docling 模型目录未找到: {docling_model_path}")

        # 检查 EasyOCR 模型
        easyocr_path = base_path / "EasyOCR"
        if easyocr_path.exists():
            log(f"✓ 找到 EasyOCR 模型目录: {easyocr_path}")

            # 检查模型子目录
            model_dir = easyocr_path / "model"
            if model_dir.exists():
                log(f"  ✓ EasyOCR model 子目录: {model_dir}")

                # 检查具体模型文件
                craft_model = model_dir / "craft_mlt_25k.pth"
                chinese_model = model_dir / "zh_sim_g2.pth"

                if craft_model.exists():
                    log(f"    ✓ CRAFT 检测模型: {craft_model}")
                else:
                    log(f"    ✗ CRAFT 检测模型未找到: {craft_model}")

                if chinese_model.exists():
                    log(f"    ✓ 中文识别模型: {chinese_model}")
                else:
                    log(f"    ✗ 中文识别模型未找到: {chinese_model}")
            else:
                log(f"  ✗ EasyOCR model 子目录未找到: {model_dir}")
        else:
            log(f"✗ EasyOCR 模型目录未找到: {easyocr_path}")

        log("模型验证完成")

    def convert_documents(self, input_doc_paths: List[Path]) -> Iterable[ConversionResult]:
        conv_results = self.doc_converter.convert_all(source=input_doc_paths)
        return conv_results

    def process_documents(self, conv_results: Iterable[ConversionResult]):
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        success_count = 0
        failure_count = 0

        for conv_res in conv_results:
            if conv_res.status == ConversionStatus.SUCCESS:
                success_count += 1
                processor = JsonReportProcessor(metadata_lookup=self.metadata_lookup,
                                                debug_data_path=self.debug_data_path)

                # Normalize the document data to ensure sequential pages
                data = conv_res.document.export_to_dict()
                normalized_data = self._normalize_page_sequence(data)

                processed_report = processor.assemble_report(conv_res, normalized_data)
                doc_filename = conv_res.input.file.stem
                if self.output_dir is not None:
                    with (self.output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
                        json.dump(processed_report, fp, indent=2, ensure_ascii=False)
                
                # 每处理完一个文档后清理显存
                self.clear_gpu_memory()
            else:
                failure_count += 1
                _log.info(f"Document {conv_res.input.file} failed to convert.")

        _log.info(f"Processed {success_count + failure_count} docs, of which {failure_count} failed")
        return success_count, failure_count

    def _normalize_page_sequence(self, data: dict) -> dict:
        """Ensure that page numbers in content are sequential by filling gaps with empty pages."""
        if 'content' not in data:
            return data

        # Create a copy of the data to modify
        normalized_data = data.copy()

        # Get existing page numbers and find max page
        existing_pages = {page['page'] for page in data['content']}
        max_page = max(existing_pages)

        # Create template for empty page
        empty_page_template = {
            "content": [],
            "page_dimensions": {}  # or some default dimensions if needed
        }

        # Create new content array with all pages
        new_content = []
        for page_num in range(1, max_page + 1):
            # Find existing page or create empty one
            page_content = next(
                (page for page in data['content'] if page['page'] == page_num),
                {"page": page_num, **empty_page_template}
            )
            new_content.append(page_content)

        normalized_data['content'] = new_content
        return normalized_data

    def clear_gpu_memory(self):
        """清理GPU显存"""
        try:
            if self.gpu_available:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if self.verbose:
                        log("已清理CUDA显存")
                elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # MPS设备清理
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                        if self.verbose:
                            log("已清理MPS显存")
        except Exception as e:
            if self.verbose:
                log(f"清理显存时出错: {str(e)}")

    def parse_and_export(self, input_doc_paths: List[Path] = None, doc_dir: Path = None):
        start_time = time.time()
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))

        total_docs = len(input_doc_paths)
        _log.info(f"Starting to process {total_docs} documents")

        conv_results = self.convert_documents(input_doc_paths)
        success_count, failure_count = self.process_documents(conv_results=conv_results)
        elapsed_time = time.time() - start_time
        
        # 处理完成后清理显存
        self.clear_gpu_memory()

        if failure_count > 0:
            error_message = f"Failed converting {failure_count} out of {total_docs} documents."
            failed_docs = "Paths of failed docs:\n" + '\n'.join(str(path) for path in input_doc_paths)
            _log.error(error_message)
            _log.error(failed_docs)
            raise RuntimeError(error_message)

        _log.info(
            f"{'#' * 50}\nCompleted in {elapsed_time:.2f} seconds. Successfully converted {success_count}/{total_docs} documents.\n{'#' * 50}")

    def parse_and_export_parallel(
            self,
            input_doc_paths: List[Path] = None,
            doc_dir: Path = None,
            optimal_workers: int = 10,
            chunk_size: int = None
    ):
        """Parse PDF files in parallel using multiple processes.

        Args:
            input_doc_paths: List of paths to PDF files to process
            doc_dir: Directory containing PDF files (used if input_doc_paths is None)
            optimal_workers: Number of worker processes to use. If None, uses CPU count.
        """
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Get input paths if not provided
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))

        total_pdfs = len(input_doc_paths)
        _log.info(f"Starting parallel processing of {total_pdfs} documents")

        cpu_count = multiprocessing.cpu_count()

        # Calculate optimal workers if not specified
        if optimal_workers is None:
            optimal_workers = min(cpu_count, total_pdfs)

        if chunk_size is None:
            # Calculate chunk size (ensure at least 1)
            chunk_size = max(1, total_pdfs // optimal_workers)

        # Split documents into chunks
        chunks = [
            input_doc_paths[i: i + chunk_size]
            for i in range(0, total_pdfs, chunk_size)
        ]

        start_time = time.time()
        processed_count = 0

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            # Schedule all tasks
            futures = [
                executor.submit(
                    _process_chunk,
                    chunk,
                    self.pdf_backend,
                    self.output_dir,
                    self.num_threads,
                    self.metadata_lookup,
                    self.debug_data_path
                )
                for chunk in chunks
            ]

            # Wait for completion and log results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    processed_count += int(result.split()[1])  # Extract number from "Processed X PDFs"
                    _log.info(f"{'#' * 50}\n{result} ({processed_count}/{total_pdfs} total)\n{'#' * 50}")
                except Exception as e:
                    _log.error(f"Error processing chunk: {str(e)}")
                    raise

        elapsed_time = time.time() - start_time
        _log.info(f"Parallel processing completed in {elapsed_time:.2f} seconds.")


class JsonReportProcessor:
    def __init__(self, metadata_lookup: dict = None, debug_data_path: Path = None):
        self.metadata_lookup = metadata_lookup or {}
        self.debug_data_path = debug_data_path

    def assemble_report(self, conv_result, normalized_data=None):
        """Assemble the report using either normalized data or raw conversion result."""
        data = normalized_data if normalized_data is not None else conv_result.document.export_to_dict()
        assembled_report = {}
        assembled_report['metainfo'] = self.assemble_metainfo(data)
        assembled_report['content'] = self.assemble_content(data)
        assembled_report['tables'] = self.assemble_tables(conv_result.document.tables, data)
        assembled_report['pictures'] = self.assemble_pictures(data)
        self.debug_data(data)
        return assembled_report

    def assemble_metainfo(self, data):
        metainfo = {}
        sha1_name = data['origin']['filename'].rsplit('.', 1)[0]
        metainfo['sha1_name'] = sha1_name
        metainfo['pages_amount'] = len(data.get('pages', []))
        metainfo['text_blocks_amount'] = len(data.get('texts', []))
        metainfo['tables_amount'] = len(data.get('tables', []))
        metainfo['pictures_amount'] = len(data.get('pictures', []))
        metainfo['equations_amount'] = len(data.get('equations', []))
        metainfo['footnotes_amount'] = len([t for t in data.get('texts', []) if t.get('label') == 'footnote'])

        # Add CSV metadata if available
        if self.metadata_lookup and sha1_name in self.metadata_lookup:
            csv_meta = self.metadata_lookup[sha1_name]
            metainfo['company_name'] = csv_meta['company_name']

        return metainfo

    def process_table(self, table_data):
        # Implement your table processing logic here
        return 'processed_table_content'

    def debug_data(self, data):
        if self.debug_data_path is None:
            return
        doc_name = data['name']
        path = self.debug_data_path / f"{doc_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def expand_groups(self, body_children, groups):
        expanded_children = []

        for item in body_children:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                if ref_type == 'groups':
                    group = groups[ref_num]
                    group_id = ref_num
                    group_name = group.get('name', '')
                    group_label = group.get('label', '')

                    for child in group['children']:
                        child_copy = child.copy()
                        child_copy['group_id'] = group_id
                        child_copy['group_name'] = group_name
                        child_copy['group_label'] = group_label
                        expanded_children.append(child_copy)
                else:
                    expanded_children.append(item)
            else:
                expanded_children.append(item)

        return expanded_children

    def _process_text_reference(self, ref_num, data):
        """Helper method to process text references and create content items.

        Args:
            ref_num (int): Reference number for the text item
            data (dict): Document data dictionary

        Returns:
            dict: Processed content item with text information
        """
        text_item = data['texts'][ref_num]
        item_type = text_item['label']
        content_item = {
            'text': text_item.get('text', ''),
            'type': item_type,
            'text_id': ref_num
        }

        # Add 'orig' field only if it differs from 'text'
        orig_content = text_item.get('orig', '')
        if orig_content != text_item.get('text', ''):
            content_item['orig'] = orig_content

        # Add additional fields if they exist
        if 'enumerated' in text_item:
            content_item['enumerated'] = text_item['enumerated']
        if 'marker' in text_item:
            content_item['marker'] = text_item['marker']

        return content_item

    def assemble_content(self, data):
        pages = {}
        # Expand body children to include group references
        body_children = data['body']['children']
        groups = data.get('groups', [])
        expanded_body_children = self.expand_groups(body_children, groups)

        # Process body content
        for item in expanded_body_children:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                if ref_type == 'texts':
                    text_item = data['texts'][ref_num]
                    content_item = self._process_text_reference(ref_num, data)

                    # Add group information if available
                    if 'group_id' in item:
                        content_item['group_id'] = item['group_id']
                        content_item['group_name'] = item['group_name']
                        content_item['group_label'] = item['group_label']

                    # Get page number from prov
                    if 'prov' in text_item and text_item['prov']:
                        page_num = text_item['prov'][0]['page_no']

                        # Initialize page if not exists
                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': text_item['prov'][0].get('bbox', {})
                            }

                        pages[page_num]['content'].append(content_item)

                elif ref_type == 'tables':
                    table_item = data['tables'][ref_num]
                    content_item = {
                        'type': 'table',
                        'table_id': ref_num
                    }

                    if 'prov' in table_item and table_item['prov']:
                        page_num = table_item['prov'][0]['page_no']

                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': table_item['prov'][0].get('bbox', {})
                            }

                        pages[page_num]['content'].append(content_item)

                elif ref_type == 'pictures':
                    picture_item = data['pictures'][ref_num]
                    content_item = {
                        'type': 'picture',
                        'picture_id': ref_num
                    }

                    if 'prov' in picture_item and picture_item['prov']:
                        page_num = picture_item['prov'][0]['page_no']

                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': picture_item['prov'][0].get('bbox', {})
                            }

                        pages[page_num]['content'].append(content_item)

        sorted_pages = [pages[page_num] for page_num in sorted(pages.keys())]
        return sorted_pages

    def assemble_tables(self, tables, data):
        assembled_tables = []
        for i, table in enumerate(tables):
            table_json_obj = table.model_dump()
            table_md = self._table_to_md(table_json_obj)
            table_html = table.export_to_html()

            table_data = data['tables'][i]
            table_page_num = table_data['prov'][0]['page_no']
            table_bbox = table_data['prov'][0]['bbox']
            table_bbox = [
                table_bbox['l'],
                table_bbox['t'],
                table_bbox['r'],
                table_bbox['b']
            ]

            # Get rows and columns from the table data structure
            nrows = table_data['data']['num_rows']
            ncols = table_data['data']['num_cols']

            ref_num = table_data['self_ref'].split('/')[-1]
            ref_num = int(ref_num)

            table_obj = {
                'table_id': ref_num,
                'page': table_page_num,
                'bbox': table_bbox,
                '#-rows': nrows,
                '#-cols': ncols,
                'markdown': table_md,
                'html': table_html,
                'json': table_json_obj
            }
            assembled_tables.append(table_obj)
        return assembled_tables

    def _table_to_md(self, table):
        # Extract text from grid cells
        table_data = []
        for row in table['data']['grid']:
            table_row = [cell['text'] for cell in row]
            table_data.append(table_row)

        # Check if the table has headers
        if len(table_data) > 1 and len(table_data[0]) > 0:
            try:
                md_table = tabulate(
                    table_data[1:], headers=table_data[0], tablefmt="github"
                )
            except ValueError:
                md_table = tabulate(
                    table_data[1:],
                    headers=table_data[0],
                    tablefmt="github",
                    disable_numparse=True,
                )
        else:
            md_table = tabulate(table_data, tablefmt="github")

        return md_table

    def assemble_pictures(self, data):
        assembled_pictures = []
        for i, picture in enumerate(data['pictures']):
            children_list = self._process_picture_block(picture, data)

            ref_num = picture['self_ref'].split('/')[-1]
            ref_num = int(ref_num)

            picture_page_num = picture['prov'][0]['page_no']
            picture_bbox = picture['prov'][0]['bbox']
            picture_bbox = [
                picture_bbox['l'],
                picture_bbox['t'],
                picture_bbox['r'],
                picture_bbox['b']
            ]

            picture_obj = {
                'picture_id': ref_num,
                'page': picture_page_num,
                'bbox': picture_bbox,
                'children': children_list,
            }
            assembled_pictures.append(picture_obj)
        return assembled_pictures

    def _process_picture_block(self, picture, data):
        children_list = []

        for item in picture['children']:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                if ref_type == 'texts':
                    content_item = self._process_text_reference(ref_num, data)

                    children_list.append(content_item)

        return children_list
