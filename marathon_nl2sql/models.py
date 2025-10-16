from pydantic import BaseModel, Field
from typing import List

class Table(BaseModel):
    """SQL数据库中的表。"""
    name: List[str] = Field(description="SQL数据库中表的名称列表。")


class QueryRequest(BaseModel):
    query: str = Field(description="自然语言查询字符串")

class QueryResponse(BaseModel):
    answer: str = Field(description="查询结果答案")
    sql_query: str = Field(description="生成的SQL查询（可选）", default=None)