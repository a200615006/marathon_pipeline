from typing import List, Dict, Any, Optional
from openai import BaseModel


class ToolRequest(BaseModel):
    """工具调用请求模型"""
    tool_name: str
    arguments: Dict[str, Any]


class UserQuery(BaseModel):
    """用户查询请求模型"""
    query: str
    max_iterations: int = 5


class ToolResponse(BaseModel):
    """工具调用响应模型"""
    success: bool
    result: str
    tool_name: str
    duration: float


class QueryResponse(BaseModel):
    """查询响应模型"""
    code: str = 0
    success: bool
    response: str
    tool_calls: List[Dict[str, Any]]
    total_iterations: int


# 请求和响应数据模型
class ChoiceQuestionRequest(BaseModel):
    segments: str
    paper: str
    id: int
    question: str
    category: str
    content: Optional[str] = None


class ChoiceQuestionResponse(BaseModel):
    segments: str
    paper: str
    id: int
    answer: str


class QAQuestionRequest(BaseModel):
    segments: str
    paper: str
    id: int
    category: str
    question: str


class QAQuestionResponse(BaseModel):
    segments: str
    paper: str
    id: int
    answer: str
