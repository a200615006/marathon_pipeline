from llama_index.core import get_response_synthesizer, PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LongContextReorder
from llama_index.core.schema import QueryBundle, NodeWithScore
from typing import List, Optional
from .retriever import Qwen3Reranker, create_parent_postprocessor
from .config import BATCH_SIZE
import time
from typing import Dict
import logging

# 基本配置
logging.basicConfig(level=logging.INFO)
log = logging.info  # 或者使用 logger
text_qa_template_str_QA = (
"你是一个仅依据给定上下文作答的助手。\n"
"上下文：\n{context_str}\n"
"要求：\n"
"1) 只使用上下文信息作答，不引入外部知识；\n"
"2) 回答尽量简洁清晰，优先≤300字符（上限≤1000字符），不加前后缀；\n"
"3) 优先使用与上下文一致的术语、数值与单位；优先给出与问题最匹配的原句或等价改写，避免冗余；\n"
"5) 如上下文无关或缺失答案，严格答：“根据已有知识无法回答”。\n"
"问题：{query_str}\n"
"回答："
)
text_qa_template_QA = PromptTemplate(text_qa_template_str_QA)


text_qa_template_str_CHOICE = (
    "上下文信息如下：\n"
    "{context_str}\n"
    "基于提供的上下文，直接回答选择题,返回答案只返回单个选项字母，如A、B、C、D等。\n"
    "选择题：{query_str}\n"
    "答案："
)

text_qa_template_CHOICE = PromptTemplate(text_qa_template_str_CHOICE)

refine_template_str = (
    "原始查询是：{query_str}\n"
    "我们已有回答：{existing_answer}\n"
    "基于以下新上下文，用中文精炼现有回答，若是问答题，问题的核心回答要放在最前边，然后是解释，确保完整性和准确性；若是选择题，则只回答选项单个字母，若上下文中没有相关答案，则严格回答“根据已有知识无法回答”：\n"
    "{context_msg}\n"
    "精炼后的回答："
)
refine_template = PromptTemplate(refine_template_str)

def create_response_synthesizer_QA():
    """创建响应合成器。"""
    return get_response_synthesizer(
        text_qa_template=text_qa_template_QA,
        refine_template=refine_template,
        response_mode="compact"
    )

def create_response_synthesizer_CHOICE():
    """创建响应合成器。"""
    return get_response_synthesizer(
        text_qa_template=text_qa_template_CHOICE,
        refine_template=refine_template,
        response_mode="compact"
    )


class DynamicQueryEngine:
    """支持动态后处理器的查询引擎"""
    
    def __init__(
        self, 
        retriever, 
        response_synthesizer, 
        reranker, 
        keep_top_k=5,
        use_parent_nodes=True,
        reorder=None
    ):
        self.retriever = retriever
        self.response_synthesizer = response_synthesizer
        self.reranker = reranker
        self.keep_top_k = keep_top_k
        self.use_parent_nodes = use_parent_nodes
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
        
        # 简洁的耗时输出
        # log(f"检索: {timing_stats['检索']:.2f}s | Rerank: {timing_stats['Rerank']:.2f}s | 生成: {timing_stats['生成回答']:.2f}s | 总计: {timing_stats['总耗时']:.2f}s")

            # 打印耗时统计
        log("\n" + "="*50)
        log("⏱️  耗时统计:")
        log("="*50)
        for step, duration in timing_stats.items():
            if step != '总耗时':
                percentage = (duration / timing_stats['总耗时']) * 100
                log(f"{step:12s}: {duration:6.3f}秒 ({percentage:5.1f}%)")
        log("-"*50)
        log(f"{'总耗时':12s}: {timing_stats['总耗时']:6.3f}秒 (100.0%)")
        log("="*50 + "\n")
    
        
        return response

def create_query_engine(hybrid_retriever, response_synthesizer, reranker):
    """创建查询引擎。"""
    longcontextreorder = LongContextReorder()
    return RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[reranker, longcontextreorder]
    )
