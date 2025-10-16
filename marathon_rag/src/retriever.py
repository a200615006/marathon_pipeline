from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever, BaseRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import torch
import copy
from .utils import clear_gpu_cache
from .config import RERANKER_MODEL, MAX_LENGTH, BATCH_SIZE

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, MetadataMode
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from typing import List, Optional, Any, Union
from pathlib import Path
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

from typing import List, Optional, Any, Dict, Tuple
import math


from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, MetadataMode
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.callbacks.schema import CBEventType, EventPayload

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt  # 正确的导入路径
from vllm.distributed.parallel_state import destroy_model_parallel
import time
from typing import Dict
import logging

# 基本配置
logging.basicConfig(level=logging.INFO)
log = logging.info  # 或者使用 logger

class Qwen3Reranker(BaseNodePostprocessor):
    """
    Qwen3-Reranker for reranking nodes based on relevance to query.
    
    Args:
        model (str): Path to the Qwen3-Reranker model.
        top_n (int): Number of nodes to return sorted by score. Defaults to 5.
        device (str, optional): Device (like "cuda", "cpu") for computation. 
            If None, checks if a GPU can be used.
        max_length (int): Maximum sequence length. Defaults to 8192.
        instruction (str, optional): Custom instruction for reranking.
        keep_retrieval_score (bool, optional): Whether to keep the retrieval score 
            in metadata. Defaults to False.
        batch_size (int): Number of query-document pairs to process at once.
            Defaults to 5. Lower values use less memory but may be slower.
        use_kv_cache (bool): Whether to use KV cache for acceleration.
            Defaults to True.
        max_kv_cache_size (int): Maximum number of batches to keep in KV cache.
            Defaults to 3. Set to 0 to disable cache limit.
        enable_gradient_checkpointing (bool): Whether to use gradient checkpointing
            to save memory. Defaults to False.
    """
    
    model: str = Field(description="Path to Qwen3-Reranker model.")
    top_n: int = Field(default=5, description="Number of nodes to return sorted by score.")
    device: Optional[str] = Field(default=None, description="Device for computation.")
    max_length: int = Field(default=8192, description="Maximum sequence length.")
    instruction: Optional[str] = Field(
        default=None, 
        description="Custom instruction for reranking."
    )
    keep_retrieval_score: bool = Field(
        default=False,
        description="Whether to keep the retrieval score in metadata.",
    )
    batch_size: int = Field(
        default=5,
        description="Number of query-document pairs to process at once.",
    )
    use_kv_cache: bool = Field(
        default=True,
        description="Whether to use KV cache for acceleration.",
    )
    max_kv_cache_size: int = Field(
        default=3,
        description="Maximum number of batches to keep in KV cache. 0 means no limit.",
    )
    enable_gradient_checkpointing: bool = Field(
        default=False,
        description="Whether to use gradient checkpointing to save memory.",
    )
    
    # 私有属性
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _device: str = PrivateAttr()
    _token_false_id: int = PrivateAttr()
    _token_true_id: int = PrivateAttr()
    _prefix: str = PrivateAttr()
    _suffix: str = PrivateAttr()
    _prefix_tokens: List[int] = PrivateAttr()
    _suffix_tokens: List[int] = PrivateAttr()
    _kv_cache_counter: int = PrivateAttr(default=0)
    
    def __init__(
        self,
        model: str,
        top_n: int = 5,
        device: Optional[str] = None,
        max_length: int = 8192,
        instruction: Optional[str] = None,
        keep_retrieval_score: bool = False,
        batch_size: int = 5,
        use_kv_cache: bool = True,
        max_kv_cache_size: int = 3,
        enable_gradient_checkpointing: bool = False,
        **kwargs
    ):
        # 先调用父类初始化，传递所有 Field 属性
        super().__init__(
            model=model,
            top_n=top_n,
            device=device,
            max_length=max_length,
            instruction=instruction,
            keep_retrieval_score=keep_retrieval_score,
            batch_size=batch_size,
            use_kv_cache=use_kv_cache,
            max_kv_cache_size=max_kv_cache_size,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            **kwargs
        )
        
        # 验证 batch_size
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        
        # 设置默认 instruction
        if self.instruction is None:
            self.instruction = "Given a web search query, retrieve relevant passages that answer the query"
        
        # 推断设备
        if self.device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device
        
        # 加载 tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model, 
            padding_side='left'
        )
        
        # 加载模型
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=torch.float16,  # 使用 float16 减少显存
                attn_implementation="flash_attention_2",
                use_cache=self.use_kv_cache  # 根据配置启用 KV cache
            ).to(self._device).eval()
            log("✓ Using flash_attention_2")
        except Exception as e:
            log(f"⚠ Flash attention not available, using default: {e}")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=torch.float16,
                use_cache=self.use_kv_cache
            ).to(self._device).eval()
        
        # 启用 gradient checkpointing 以节省显存
        if self.enable_gradient_checkpointing:
            self._model.gradient_checkpointing_enable()
            log("✓ Gradient checkpointing enabled")
        
        # 获取 yes/no token ids
        self._token_false_id = self._tokenizer.convert_tokens_to_ids("no")
        self._token_true_id = self._tokenizer.convert_tokens_to_ids("yes")
        
        # 定义前缀和后缀
        self._prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self._suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self._prefix_tokens = self._tokenizer.encode(self._prefix, add_special_tokens=False)
        self._suffix_tokens = self._tokenizer.encode(self._suffix, add_special_tokens=False)
        
        # 初始化 KV cache 计数器
        self._kv_cache_counter = 0
    
    @classmethod
    def class_name(cls) -> str:
        """返回类名,用于序列化"""
        return "Qwen3Reranker"
    
    def _selective_cache_clear(self):
        """
        选择性清理缓存:
        - 只清理输入张量,不清理模型的 KV cache
        - 定期清理以防止显存累积
        """
        if self._device.startswith("cuda"):
            # 只清空未使用的缓存,不影响已分配的 KV cache
            torch.cuda.empty_cache()
            
            # 每处理 max_kv_cache_size 个批次后,进行一次完整清理
            self._kv_cache_counter += 1
            if self.max_kv_cache_size > 0 and self._kv_cache_counter >= self.max_kv_cache_size:
                # 清理模型的 past_key_values (KV cache)
                if hasattr(self._model, 'clear_cache'):
                    self._model.clear_cache()
                
                # 强制垃圾回收
                gc.collect()
                torch.cuda.synchronize()
                
                # 重置计数器
                self._kv_cache_counter = 0
                log(f"✓ Periodic cache clear after {self.max_kv_cache_size} batches")
    
    def _format_instruction(self, query: str, doc: str) -> str:
        """格式化输入文本"""
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"
    
    def _process_inputs(self, pairs: List[str]):
        """处理输入对"""
        inputs = self._tokenizer(
            pairs, 
            padding=False, 
            truncation='longest_first',
            return_attention_mask=False, 
            max_length=self.max_length - len(self._prefix_tokens) - len(self._suffix_tokens)
        )
        
        # 添加前缀和后缀
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self._prefix_tokens + ele + self._suffix_tokens
        
        # 填充
        inputs = self._tokenizer.pad(
            inputs, 
            padding=True, 
            return_tensors="pt", 
            max_length=self.max_length
        )
        
        # 移动到设备
        for key in inputs:
            inputs[key] = inputs[key].to(self._device)
        
        return inputs
    
    @torch.no_grad()
    def _compute_scores_batch(self, pairs: List[str], past_key_values=None) -> tuple:
        """
        计算一批 pairs 的相关性分数,支持 KV cache
        
        Args:
            pairs: 格式化后的 query-document 对列表
            past_key_values: 上一批次的 KV cache (如果启用)
            
        Returns:
            (scores, new_past_key_values) 元组
        """
        inputs = self._process_inputs(pairs)
        
        try:
            # 如果使用 KV cache 且有 past_key_values,传入模型
            model_kwargs = {}
            if self.use_kv_cache and past_key_values is not None:
                model_kwargs['past_key_values'] = past_key_values
                model_kwargs['use_cache'] = True
            
            outputs = self._model(**inputs, **model_kwargs)
            batch_scores = outputs.logits[:, -1, :]
            
            true_vector = batch_scores[:, self._token_true_id]
            false_vector = batch_scores[:, self._token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
            
            # 返回分数和新的 KV cache
            new_past_key_values = outputs.past_key_values if self.use_kv_cache else None
            
            return scores, new_past_key_values
            
        finally:
            # 只删除输入张量,保留 KV cache
            del inputs
            # 选择性清理缓存
            self._selective_cache_clear()
    
    def _compute_scores(self, pairs: List[str]) -> List[float]:
        """
        分批计算所有 pairs 的相关性分数,使用 KV cache 加速
        
        Args:
            pairs: 格式化后的 query-document 对列表
            
        Returns:
            所有 pairs 的分数列表
        """
        all_scores = []
        total_pairs = len(pairs)
        past_key_values = None  # 用于存储 KV cache
        
        # 分批处理
        for i in range(0, total_pairs, self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            
            # 注意:对于 reranker,每个 batch 的 query-doc 对是独立的
            # 所以这里不传递 past_key_values,每个 batch 独立计算
            # 如果你的场景是同一个 query 对多个 doc,可以复用 query 的 KV cache
            batch_scores, _ = self._compute_scores_batch(batch_pairs, past_key_values=None)
            all_scores.extend(batch_scores)
            
            # 打印进度
            if total_pairs > self.batch_size:
                processed = min(i + self.batch_size, total_pairs)
                log(f"Reranking progress: {processed}/{total_pairs} pairs processed")
        
        return all_scores
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        重排序节点(必须实现的抽象方法)
        
        Args:
            nodes: 待重排序的节点列表
            query_bundle: 查询信息
            
        Returns:
            重排序后的节点列表
        """
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        
        if len(nodes) == 0:
            return []
        
        # 重置 KV cache 计数器(新的查询开始)
        self._kv_cache_counter = 0
        
        try:
            # 准备查询-文档对
            query_str = query_bundle.query_str
            query_and_nodes = [
                (
                    query_str,
                    node.node.get_content(metadata_mode=MetadataMode.EMBED),
                )
                for node in nodes
            ]
            
            # 格式化输入
            pairs = [
                self._format_instruction(query, doc) 
                for query, doc in query_and_nodes
            ]
            
            # 使用 callback manager 记录事件
            with self.callback_manager.event(
                CBEventType.RERANKING,
                payload={
                    EventPayload.NODES: nodes,
                    EventPayload.MODEL_NAME: self.model,
                    EventPayload.QUERY_STR: query_str,
                    EventPayload.TOP_K: self.top_n,
                },
            ) as event:
                # 分批处理并计算分数
                scores = self._compute_scores(pairs)
                
                assert len(scores) == len(nodes), \
                    f"Score count mismatch: got {len(scores)} scores for {len(nodes)} nodes"
                
                # 更新节点分数
                for node, score in zip(nodes, scores):
                    if self.keep_retrieval_score:
                        node.node.metadata["retrieval_score"] = node.score
                    node.score = float(score)
                
                # 按分数排序并返回 top_n
                new_nodes = sorted(
                    nodes, 
                    key=lambda x: -x.score if x.score else 0
                )[: self.top_n]
                
                # 记录结果
                event.on_end(payload={EventPayload.NODES: new_nodes})
            
            return new_nodes
        
        finally:
            # 完成一次完整的 reranking 后,清理一次缓存
            if self._device.startswith("cuda"):
                # 清理模型的 KV cache
                if hasattr(self._model, 'clear_cache'):
                    self._model.clear_cache()
                
                torch.cuda.empty_cache()
                gc.collect()
                log("✓ Cache cleared after reranking completion")

class Qwen3Reranker_vllm(BaseNodePostprocessor):
    """
    Qwen3-Reranker with vLLM acceleration for high-performance reranking.
    
    Args:
        model (str): Path to the Qwen3-Reranker model. Defaults to "Qwen/Qwen3-Reranker-4B".
        top_n (int): Number of nodes to return sorted by score. Defaults to 5.
        max_length (int): Maximum sequence length. Defaults to 8192.
        instruction (str, optional): Custom instruction for reranking.
        keep_retrieval_score (bool): Whether to keep the retrieval score in metadata. 
            Defaults to False.
        tensor_parallel_size (int, optional): Number of GPUs for tensor parallelism.
            If None, uses all available GPUs.
        gpu_memory_utilization (float): GPU memory utilization ratio. Defaults to 0.8.
        enable_prefix_caching (bool): Whether to enable prefix caching in vLLM.
            Defaults to True for better performance.
        batch_size (int): Number of query-document pairs to process at once.
            Defaults to 32 (vLLM can handle larger batches efficiently).
        max_model_len (int): Maximum model length for vLLM. Defaults to 10000.
    """
    
    model: str = Field(
        default="Qwen/Qwen3-Reranker-4B",
        description="Path to Qwen3-Reranker model."
    )
    top_n: int = Field(
        default=5, 
        description="Number of nodes to return sorted by score."
    )
    max_length: int = Field(
        default=8192, 
        description="Maximum sequence length."
    )
    instruction: Optional[str] = Field(
        default=None, 
        description="Custom instruction for reranking."
    )
    keep_retrieval_score: bool = Field(
        default=False,
        description="Whether to keep the retrieval score in metadata.",
    )
    tensor_parallel_size: Optional[int] = Field(
        default=None,
        description="Number of GPUs for tensor parallelism.",
    )
    gpu_memory_utilization: float = Field(
        default=0.8,
        description="GPU memory utilization ratio.",
    )
    enable_prefix_caching: bool = Field(
        default=True,
        description="Whether to enable prefix caching in vLLM.",
    )
    batch_size: int = Field(
        default=32,
        description="Number of query-document pairs to process at once.",
    )
    max_model_len: int = Field(
        default=10000,
        description="Maximum model length for vLLM.",
    )
    
    # 私有属性
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _sampling_params: Any = PrivateAttr()
    _token_false_id: int = PrivateAttr()
    _token_true_id: int = PrivateAttr()
    _suffix: str = PrivateAttr()
    _suffix_tokens: List[int] = PrivateAttr()
    _system_message: str = PrivateAttr()
    
    def __init__(
        self,
        model: str = "Qwen/Qwen3-Reranker-4B",
        top_n: int = 5,
        max_length: int = 8192,
        instruction: Optional[str] = None,
        keep_retrieval_score: bool = False,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.8,
        enable_prefix_caching: bool = True,
        batch_size: int = 32,
        max_model_len: int = 10000,
        **kwargs
    ):
        # 调用父类初始化
        super().__init__(
            model=model,
            top_n=top_n,
            max_length=max_length,
            instruction=instruction,
            keep_retrieval_score=keep_retrieval_score,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            batch_size=batch_size,
            max_model_len=max_model_len,
            **kwargs
        )
        
        # 验证参数
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}"
            )
        
        # 设置默认 instruction
        if self.instruction is None:
            self.instruction = "Given a web search query, retrieve relevant passages that answer the query"
        
        # 设置系统消息
        self._system_message = (
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\"."
        )
        
        # 推断 tensor_parallel_size
        if self.tensor_parallel_size is None:
            self.tensor_parallel_size = torch.cuda.device_count()
            if self.tensor_parallel_size == 0:
                self.tensor_parallel_size = 1
                log("⚠ No GPU detected, using CPU (will be slow)")
        
        log(f"✓ Initializing vLLM with {self.tensor_parallel_size} GPU(s)")
        
        # 加载 tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._tokenizer.padding_side = "left"
        self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # 初始化 vLLM 模型
        self._model = LLM(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            enable_prefix_caching=self.enable_prefix_caching,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
        )
        
        log(f"✓ vLLM model loaded successfully")
        log(f"  - Prefix caching: {self.enable_prefix_caching}")
        log(f"  - GPU memory utilization: {self.gpu_memory_utilization}")
        log(f"  - Max model length: {self.max_model_len}")
        
        # 获取 yes/no token ids
        self._token_true_id = self._tokenizer("yes", add_special_tokens=False).input_ids[0]
        self._token_false_id = self._tokenizer("no", add_special_tokens=False).input_ids[0]
        
        # 定义后缀
        self._suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self._suffix_tokens = self._tokenizer.encode(self._suffix, add_special_tokens=False)
        
        # 配置采样参数
        self._sampling_params = SamplingParams(
            temperature=0,  # 确定性输出
            max_tokens=1,   # 只需要一个 token (yes/no)
            logprobs=20,    # 返回 top-20 logprobs
            allowed_token_ids=[self._token_true_id, self._token_false_id],  # 只允许 yes/no
        )
        
        log(f"✓ Qwen3Reranker_vllm initialized successfully")
    
    @classmethod
    def class_name(cls) -> str:
        """返回类名,用于序列化"""
        return "Qwen3Reranker_vllm"
    
    def _format_instruction(self, query: str, doc: str) -> List[Dict[str, str]]:
        """
        格式化输入为 chat 格式
        
        Args:
            query: 查询文本
            doc: 文档文本
            
        Returns:
            Chat 格式的消息列表
        """
        return [
            {"role": "system", "content": self._system_message},
            {
                "role": "user", 
                "content": f"<Instruct>: {self.instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"
            }
        ]
    
    def _process_inputs(self, pairs: List[tuple]) -> List[TokensPrompt]:
        """
        处理输入对,转换为 vLLM 所需格式
        
        Args:
            pairs: (query, doc) 元组列表
            
        Returns:
            TokensPrompt 列表
        """
        # 格式化为 chat 消息
        messages = [
            self._format_instruction(query, doc) 
            for query, doc in pairs
        ]
        
        # 应用 chat template 并 tokenize
        tokenized_messages = self._tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=False,
            enable_thinking=False  # Qwen3 特有参数
        )
        
        # 截断并添加后缀
        max_len = self.max_length - len(self._suffix_tokens)
        processed_messages = [
            ele[:max_len] + self._suffix_tokens 
            for ele in tokenized_messages
        ]
        
        # 转换为 TokensPrompt 格式
        prompts = [
            TokensPrompt(prompt_token_ids=tokens) 
            for tokens in processed_messages
        ]
        
        return prompts
    
    def _compute_scores_batch(self, pairs: List[tuple]) -> List[float]:
        """
        使用 vLLM 批量计算相关性分数
        
        Args:
            pairs: (query, doc) 元组列表
            
        Returns:
            分数列表
        """
        # 处理输入
        prompts = self._process_inputs(pairs)
        
        # 使用 vLLM 生成
        outputs = self._model.generate(
            prompts, 
            self._sampling_params, 
            use_tqdm=False
        )
        
        # 计算分数
        scores = []
        for output in outputs:
            # 获取最后一个 token 的 logprobs
            final_logits = output.outputs[0].logprobs[-1]
            
            # 提取 yes/no 的 logprob
            if self._token_true_id not in final_logits:
                true_logit = -10  # 极小值
            else:
                true_logit = final_logits[self._token_true_id].logprob
            
            if self._token_false_id not in final_logits:
                false_logit = -10
            else:
                false_logit = final_logits[self._token_false_id].logprob
            
            # 计算归一化分数
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            
            scores.append(score)
        
        return scores
    
    def _compute_scores(self, pairs: List[tuple]) -> List[float]:
        """
        分批计算所有 pairs 的相关性分数
        
        Args:
            pairs: (query, doc) 元组列表
            
        Returns:
            所有 pairs 的分数列表
        """
        all_scores = []
        total_pairs = len(pairs)
        
        # 分批处理
        for i in range(0, total_pairs, self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            batch_scores = self._compute_scores_batch(batch_pairs)
            all_scores.extend(batch_scores)
            
            # 打印进度
            if total_pairs > self.batch_size:
                processed = min(i + self.batch_size, total_pairs)
                log(f"Reranking progress: {processed}/{total_pairs} pairs processed")
        
        return all_scores
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        重排序节点(必须实现的抽象方法)
        
        Args:
            nodes: 待重排序的节点列表
            query_bundle: 查询信息
            
        Returns:
            重排序后的节点列表
        """
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        
        if len(nodes) == 0:
            return []
        
        # 准备查询-文档对
        query_str = query_bundle.query_str
        pairs = [
            (
                query_str,
                node.node.get_content(metadata_mode=MetadataMode.EMBED),
            )
            for node in nodes
        ]
        
        # 使用 callback manager 记录事件
        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            # 计算分数
            scores = self._compute_scores(pairs)
            
            assert len(scores) == len(nodes), \
                f"Score count mismatch: got {len(scores)} scores for {len(nodes)} nodes"
            
            # 更新节点分数
            for node, score in zip(nodes, scores):
                if self.keep_retrieval_score:
                    node.node.metadata["retrieval_score"] = node.score
                node.score = float(score)
            
            # 按分数排序并返回 top_n
            new_nodes = sorted(
                nodes, 
                key=lambda x: -x.score if x.score else 0
            )[: self.top_n]
            
            # 记录结果
            event.on_end(payload={EventPayload.NODES: new_nodes})
        
        return new_nodes
    
    def __del__(self):
        """清理资源"""
        try:
            # 销毁 vLLM 的分布式环境
            destroy_model_parallel()
            
            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 强制垃圾回收
            gc.collect()
            
            log("✓ Qwen3Reranker_vllm resources cleaned up")
        except Exception as e:
            log(f"⚠ Error during cleanup: {e}")




# ==================== 1. 节点分割器 ====================
class NodeSplitter:
    """将长节点分割成多个子节点,保持父子关系"""
    
    def __init__(self, chunk_size: int = 512, overlap_ratio: float = 0.1):
        """
        Args:
            chunk_size: 子节点的目标长度
            overlap_ratio: 重叠比例 (0.1 表示 10%)
        """
        self.chunk_size = chunk_size
        self.overlap_size = int(chunk_size * overlap_ratio)
            
    def split_node(self, node: NodeWithScore, parent_id: str = None) -> List[NodeWithScore]:
        """
        将单个节点分割成多个子节点
        
        Args:
            node: 原始节点
            parent_id: 父节点ID (如果为None,使用node.node.node_id)
            
        Returns:
            子节点列表,每个子节点都保留父节点引用
        """
        text = node.node.text
        text_length = len(text)
        
        # 🔥 定义要排除的元数据键(不传给LLM)
        excluded_llm_keys = [
            'category_depth', 'languages', 'filetype', 'last_modified',
            'parent_node_id', 'chunk_index', 'is_child_node', 
            'parent_text_length', 'chunk_start', 'chunk_end',
            'matched_children'  # 🔥 新增:排除子节点匹配信息
        ]
        
        # 如果文本长度小于chunk_size,直接返回原节点
        if text_length <= self.chunk_size:
            # 添加父节点ID到metadata
            node.node.metadata['parent_node_id'] = parent_id or node.node.node_id
            node.node.metadata['is_child_node'] = False
            # 🔥 配置排除的元数据
            node.node.excluded_llm_metadata_keys = excluded_llm_keys
            return [node]
        
        parent_node_id = parent_id or node.node.node_id
        child_nodes = []
        start = 0
        chunk_index = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk_text = text[start:end]
            
            # 创建子节点
            child_node = TextNode(
                text=chunk_text,
                metadata={
                    **node.node.metadata,  # 继承父节点的metadata
                    'parent_node_id': parent_node_id,
                    'chunk_index': chunk_index,
                    'is_child_node': True,
                    'parent_text_length': text_length,
                    'chunk_start': start,
                    'chunk_end': end
                },
                excluded_embed_metadata_keys=node.node.excluded_embed_metadata_keys,
                # 🔥 配置排除的元数据键
                excluded_llm_metadata_keys=excluded_llm_keys,
            )
            
            # 保持原始评分
            child_node_with_score = NodeWithScore(
                node=child_node,
                score=node.score
            )
            
            child_nodes.append(child_node_with_score)
            
            # 计算下一个起点 (带重叠)
            start += (self.chunk_size - self.overlap_size)
            chunk_index += 1
        
        return child_nodes
    
    def split_nodes(self, nodes: List[NodeWithScore]) -> tuple[List[NodeWithScore], Dict[str, NodeWithScore]]:
        """
        批量分割节点
        
        Returns:
            (子节点列表, 父节点映射字典)
        """
        all_child_nodes = []
        parent_node_map = {}  # parent_node_id -> 原始父节点
        
        for node in nodes:
            parent_id = node.node.node_id
            parent_node_map[parent_id] = node  # 保存原始父节点
            
            child_nodes = self.split_node(node, parent_id)
            all_child_nodes.extend(child_nodes)
        
        return all_child_nodes, parent_node_map


# ==================== 2. 子节点到父节点的后处理器 ====================
class ChildToParentPostprocessor(BaseNodePostprocessor):
    """
    将rerank后的子节点还原为父节点
    策略: 如果多个子节点来自同一父节点,取最高分的子节点分数作为父节点分数
    """
    
    # 使用 Pydantic 的方式声明字段
    parent_node_map: Dict[str, Any] = {}
    keep_top_k: int = 5
    
    def __init__(self, parent_node_map: Dict[str, NodeWithScore], keep_top_k: int = 5, **kwargs):
        """
        Args:
            parent_node_map: 父节点ID到父节点的映射
            keep_top_k: 最终保留的父节点数量
        """
        # 使用 Pydantic 的初始化方式
        super().__init__(
            parent_node_map=parent_node_map,
            keep_top_k=keep_top_k,
            **kwargs
        )
    
    def _postprocess_nodes(
        self, 
        nodes: List[NodeWithScore], 
        query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """
        将子节点还原为父节点
        
        评分排序逻辑:
        1. 按父节点分组所有子节点
        2. 每个父节点的得分 = 其所有子节点的最高分
        3. 按父节点得分降序排序
        4. 返回前 keep_top_k 个父节点
        """
        # 按父节点ID分组,记录每个父节点的最高分数
        parent_scores: Dict[str, float] = {}
        parent_child_nodes: Dict[str, List[NodeWithScore]] = {}
        
        for node in nodes:
            parent_id = node.node.metadata.get('parent_node_id')
            
            if not parent_id:
                # 如果没有父节点ID,说明是原始节点,直接保留
                parent_scores[node.node.node_id] = node.score
                parent_child_nodes[node.node.node_id] = [node]
                continue
            
            # 记录最高分数
            if parent_id not in parent_scores:
                parent_scores[parent_id] = node.score
                parent_child_nodes[parent_id] = [node]
            else:
                # 取最高分
                parent_scores[parent_id] = max(parent_scores[parent_id], node.score)
                parent_child_nodes[parent_id].append(node)
        
        # 🔥 定义要排除的元数据键(不传给LLM)
        excluded_llm_keys = [
            'category_depth', 'languages', 'filetype', 'last_modified',
            'parent_node_id', 'chunk_index', 'is_child_node', 
            'parent_text_length', 'chunk_start', 'chunk_end',
            'matched_children'  # 排除子节点匹配信息
        ]
        
        # 构建父节点列表
        parent_nodes = []
        for parent_id, score in parent_scores.items():
            if parent_id in self.parent_node_map:
                # 使用保存的原始父节点
                parent_node = copy.deepcopy(self.parent_node_map[parent_id])
                parent_node.score = score  # 🔥 设置为所有子节点的最高分
                
                # 🔥 配置排除的元数据键
                parent_node.node.excluded_llm_metadata_keys = excluded_llm_keys
                
                # # 🔥 可选: 记录匹配的子节点信息(用于调试,但不会传给LLM)
                # child_info = [
                #     {
                #         'chunk_index': n.node.metadata.get('chunk_index'),
                #         'score': n.score,
                #         'text_preview': n.node.text[:50] + '...'
                #     }
                #     for n in sorted(parent_child_nodes[parent_id], key=lambda x: x.score, reverse=True)
                # ]
                # parent_node.node.metadata['matched_children'] = child_info
                
                parent_nodes.append(parent_node)
            else:
                # 如果找不到父节点,使用第一个子节点(不应该发生)
                log(f"警告: 找不到父节点 {parent_id}, 使用子节点代替")
                fallback_node = parent_child_nodes[parent_id][0]
                fallback_node.node.excluded_llm_metadata_keys = excluded_llm_keys
                parent_nodes.append(fallback_node)
        
        # 🔥 按分数降序排序并返回top_k
        parent_nodes.sort(key=lambda x: x.score, reverse=True)
        return parent_nodes[:self.keep_top_k]
    
    class Config:
        arbitrary_types_allowed = True  # 允许任意类型


# ==================== 3. 自定义检索器包装器 ====================
class SplitNodeRetriever(BaseRetriever):
    """
    包装原始检索器,自动处理节点分割
    """
    
    def __init__(
        self, 
        base_retriever: BaseRetriever,
        chunk_size: int = 512,
        overlap_ratio: float = 0.1
    ):
        """
        Args:
            base_retriever: 原始混合检索器
            chunk_size: 子节点大小
            overlap_ratio: 重叠比例
        """
        super().__init__()
        self.base_retriever = base_retriever
        self.node_splitter = NodeSplitter(chunk_size, overlap_ratio)
        self.parent_node_map = {}
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        检索并分割节点
        """
        # 1. 使用原始检索器检索
        nodes = self.base_retriever.retrieve(query_bundle)
        
        # 2. 分割节点
        child_nodes, self.parent_node_map = self.node_splitter.split_nodes(nodes)
        
        log(f"原始节点数: {len(nodes)}, 分割后子节点数: {len(child_nodes)}")
        
        return child_nodes
    
    def get_parent_node_map(self) -> Dict[str, NodeWithScore]:
        """获取父节点映射,供后处理器使用"""
        return self.parent_node_map


def create_parent_postprocessor(retriever: SplitNodeRetriever, keep_top_k: int = 5):
    """动态创建父节点后处理器"""
    return ChildToParentPostprocessor(
        parent_node_map=retriever.get_parent_node_map(),
        keep_top_k=keep_top_k
    )

class DynamicQueryEngine:
    """支持动态后处理器的查询引擎"""
    
    def __init__(
        self, 
        retriever, 
        response_synthesizer, 
        reranker, 
        keep_top_k=5,
        use_parent_nodes=True,  # 🔥 新增开关
        reorder=None
    ):
        self.retriever = retriever
        self.response_synthesizer = response_synthesizer
        self.reranker = reranker
        self.keep_top_k = keep_top_k
        self.use_parent_nodes = use_parent_nodes  # 🔥 保存开关
        self.reorder = reorder

    def longcontext_postprocess_nodes(
        self,
        nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        reordered_nodes: List[NodeWithScore] = []
        ordered_nodes: List[NodeWithScore] = sorted(
            nodes, key=lambda x: x.score if x.score is not None else 0
        )
        for i, node in enumerate(ordered_nodes):
            if i % 2 == 0:
                reordered_nodes.insert(0, node)
            else:
                reordered_nodes.append(node)
        return reordered_nodes
    
    def query(self, query_str: str):
        from llama_index.core.schema import QueryBundle
        
        # 记录总开始时间
        total_start = time.time()
        timing_stats: Dict[str, float] = {}
        
        # 1. 检索 (自动分割节点)
        retrieval_start = time.time()
        query_bundle = QueryBundle(query_str=query_str)
        nodes = self.retriever.retrieve(query_bundle)
        timing_stats['检索'] = time.time() - retrieval_start
        
        # 2. Rerank子节点
        rerank_start = time.time()
        reranked_nodes = self.reranker.postprocess_nodes(nodes, query_bundle)
        timing_stats['Rerank'] = time.time() - rerank_start
        
        # 3. 根据开关决定是否还原父节点
        parent_start = time.time()
        if self.use_parent_nodes:
            parent_postprocessor = create_parent_postprocessor(
                self.retriever, 
                keep_top_k=self.keep_top_k
            )
            final_nodes = parent_postprocessor.postprocess_nodes(reranked_nodes, query_bundle)
            timing_stats['还原父节点'] = time.time() - parent_start
        else:
            final_nodes = reranked_nodes[:self.keep_top_k]
            timing_stats['截取节点'] = time.time() - parent_start
        
        # 4. Reorder (如果启用)
        if self.reorder:
            reorder_start = time.time()
            final_nodes = self.longcontext_postprocess_nodes(final_nodes)
            timing_stats['Reorder'] = time.time() - reorder_start
        
        # 5. 生成回答
        synthesis_start = time.time()
        response = self.response_synthesizer.synthesize(
            query=query_str,
            nodes=final_nodes
        )
        timing_stats['生成回答'] = time.time() - synthesis_start
        
        # 计算总耗时
        timing_stats['总耗时'] = time.time() - total_start
        log('a fking test')
        
        # 简洁的耗时输出
        log(f"检索: {timing_stats['检索']:.2f}s | Rerank: {timing_stats['Rerank']:.2f}s | 生成: {timing_stats['生成回答']:.2f}s | 总计: {timing_stats['总耗时']:.2f}s")
        
        return response