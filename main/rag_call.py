import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date

import requests

from config import RAG_URL
import logging


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RagTool:

    @classmethod
    def call(cls, question: str, content: str) -> Dict[str, Any]:
        logger.info(f"start to call rag | question={question} | content={content}")
        call_start = time.time()
        if content:
            function_args = {
                "question": question,
                "content": content
            }
        else:
            function_args = {
                "question": question
            }

        url = RAG_URL
        try:
            response = requests.post(url, json=function_args, timeout=120)
            # 统一判断：只要不是200状态码就认为是异常
            if response.status_code == 200:
                print(f"rag answer={response.json()["answer"]}")
                return {"success": True, "result": response.json()["answer"]}
            else:
                return {"success": False, "result": ""}
        except Exception as e:
            # 所有异常统一处理
            return {"success": False, "result": ""}
