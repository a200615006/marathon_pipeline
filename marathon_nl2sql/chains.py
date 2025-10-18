from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, \
    FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter

from config import OPENAI_API_KEY, OPENAI_API_BASE, MODEL_NAME, TEMPERATURE, DATABASE_URI, TABLE_DESCRIPTIONS_CSV
from utils import clean_sql_query, get_table_details
from models import Table
from typing import List

import logging
import time

from config import LOG_FILE, LOG_LEVEL

# 配置日志（模块加载时初始化）
logging.basicConfig(
    filename=LOG_FILE,
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"  # 追加模式
)
logger = logging.getLogger(__name__)

# 初始化数据库和LLM
db = SQLDatabase.from_uri(DATABASE_URI)
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE
)

# 执行查询工具
execute_query = QuerySQLDataBaseTool(db=db)

# 回答prompt
answer_prompt = PromptTemplate.from_template(
    """给定以下用户问题、对应的SQL查询和SQL结果，请回答用户问题。

    问题: {question}
    SQL 查询: {query}
    SQL 结果: {result}
    答案: """
)

rephrase_answer = answer_prompt | llm | StrOutputParser()

# 示例
examples = [
    {
        "input": "Get the highest payment amount made by any customer.",
        "query": "SELECT MAX(amount) FROM payments;"
    }
]

# Few-shot prompt
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    input_variables=["input", "top_k"],
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "您是MySQL专家。根据输入问题，创建语法正确的MySQL查询来运行。除非另有指定。\n\n以下是相关表信息：{table_info}\n\n下面是一些问题及其对应SQL查询的示例。"),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# 生成查询链
generate_query = create_sql_query_chain(llm, db, final_prompt)

# 表选择prompt
table_details_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """返回所有可能与用户问题相关的SQL表名。
                      表如下：

                      {table_details}

                      请记住包括所有潜在相关的表，即使你不确定它们是否需要。
                      请以JSON格式提供响应，其中包含键'name'，值为表名列表。"""),
        ("human", "{question}")
    ]
)

structured_llm = llm.with_structured_output(Table)

table_chain = table_details_prompt | structured_llm


def get_tables(table_response: Table) -> List[str]:
    """
    从Table对象提取表名列表。

    Args:
        table_response (Table): Pydantic Table对象。

    Returns:
        List[str]: 表名列表。
    """
    return table_response.name


select_table = {"question": itemgetter("question"),
                "table_details": itemgetter("table_details")} | table_chain | get_tables

TABLE_DETAILS = get_table_details(TABLE_DESCRIPTIONS_CSV)

def create_nl2sql_chain():
    def log_step(step_name, func):
        def wrapper(input):
            logger.info(f"开始 {step_name}...")
            start_time = time.time()
            output = func(input)
            end_time = time.time()
            logger.info(f"{step_name} 完成，耗时: {end_time - start_time:.4f} 秒")
            return output
        return wrapper
    # 定义一个可运行的链，逐步 assign 并在最后返回所需字段
    chain = (
            RunnablePassthrough.assign(table_names_to_use=select_table) |
            RunnablePassthrough.assign(query=generate_query | RunnableLambda(clean_sql_query)) |
            RunnablePassthrough.assign(result=itemgetter("query") | execute_query)
    )



    final_chain = chain | (lambda output: {
        "sql_query": output["query"],
        "answer": rephrase_answer.invoke({
            "question": output["question"],
            "query": output["query"],
            "result": output["result"]
        })
    })

    return final_chain

def run_nl2sql_query(query: str) -> dict:
    logger.info(f"开始处理查询: {query}")
    overall_start = time.time()
    chain = create_nl2sql_chain()
    # 单次 invoke，返回 dict
    result = chain.invoke({"question": query, "table_details": TABLE_DETAILS})
    overall_end = time.time()
    logger.info(f"查询处理完成，总耗时: {overall_end - overall_start:.4f} 秒")
    logger.info(f"查询结果:{result["answer"]}, 生成的sql:{result["sql_query"]}")
    return {
        "answer": result["answer"],
        "sql_query": result["sql_query"]
    }

def run_nl2sql_query_from_main(query: str) -> dict:
    """
    处理自然语言查询并返回答案和SQL查询。
    
    Args:
        query (str): 自然语言查询字符串。
    
    Returns:
        dict: 包含答案和SQL查询的字典。
    """
    chain = create_nl2sql_chain()
    result = chain.invoke({"question": query, "table_details": TABLE_DETAILS})
    return {
        "answer": result["answer"],
        "sql_query": result["sql_query"]
    }