import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date

import requests

from config import NL2SQL_URL

class Nl2sqlTool:

    @classmethod
    def call(cls, query: str) -> Dict[str, Any]:
        call_start = time.time()
        function_args = {
            "query": query
        }
        url = NL2SQL_URL
        try:
            response = requests.post(url, json=function_args, timeout=30)
            # 统一判断：只要不是200状态码就认为是异常
            if response.status_code == 200:
                print(f"nl2sql sql={response.json()["sql_query"]}")
                return {"success": True, "result": response.json()["answer"]}
            else:
                return {"success": False, "result": ""}
        except Exception as e:
            # 所有异常统一处理
            return {"success": False, "result": ""}
