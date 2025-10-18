from typing import Sequence
import requests
import re  # 用于解析示例
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken

# 自定义 RAG 代理（模拟检索服务，去除历史）
class RAGAgent(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, description="执行 RAG 检索服务，获取相关数据。")

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        query = messages[-1].content  # 直接从传入消息获取最新输入
        # 模拟调用 RAG 服务（替换为实际 API）
        # 示例：假设查询腾讯财报，返回收入数据
        response = requests.post("http://your-rag-service/api/retrieve", json={"query": query})  # 实际调用
        result = response.json().get("retrieved_content", "模拟 RAG 结果：腾讯 2025 Q2 收入 1845.04 亿元人民币。")
        response_message = TextMessage(content=result, source=self.name)
        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass  # 无需历史，无操作

# 自定义 NL2SQL 代理（模拟数据库查询服务，去除历史）
class NL2SQLAgent(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, description="执行 NL2SQL 查询服务，从数据库获取数据。")

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        input_data = messages[-1].content  # 直接从传入消息获取
        # 模拟调用 NL2SQL 服务（替换为实际 FastAPI API）
        response = requests.post("http://your-nl2sql-service/api/convert", json={"query": input_data})
        sql = response.json().get("sql", "SELECT * FROM table")  # 实际 SQL
        db_result = "模拟数据库结果"  # 替换为 execute_sql(sql)
        response_message = TextMessage(content=str(db_result), source=self.name)
        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

# 自定义 MCP 代理（模拟自定义处理工具，去除历史）
class MCPAgent(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, description="执行 MCP 处理工具，进行数据转换或计算。")

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        input_data = messages[-1].content  # 直接从传入消息获取
        # 模拟调用 MCP 服务（替换为实际 API）
        # 示例：汇率换算
        income_rmb = float(re.search(r'(\d+\.?\d*) 亿元人民币', input_data).group(1)) if re.search(r'(\d+\.?\d*) 亿元人民币', input_data) else 0
        exchange_rate = 0.1202  # 模拟汇率
        result = f"模拟 MCP 结果：换算欧元 {income_rmb * exchange_rate:.2f} 亿元。"
        response_message = TextMessage(content=result, source=self.name)
        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

# 自定义整合代理（合成最终输出，去除历史）
class IntegratorAgent(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, description="整合所有结果，生成最终答案。")

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        # 收集传入消息结果（假设所有相关消息已传入）
        results = [msg.content for msg in messages if isinstance(msg, TextMessage)]
        final_answer = "最终整合答案： " + " ".join(results[-3:])  # 示例整合最后三个结果
        response_message = TextMessage(content=final_answer, source=self.name)
        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass