import re
from typing import Dict, Optional

# 模拟信用卡账单数据（实际应用中可替换为数据库或API调用）
MOCK_CREDIT_CARD_DATA: Dict[str, Dict] = {

"2024-09":{
  "card_number": "6211****1111",
  "cardholder_name": "张三",
  "bank": "中国银行",
  "bill_month": "2024-09",
  "total_amount": 19345.50,
  "minimum_payment": 1568.05,
  "payment_status": "unpaid",
  "due_date": "2025-10-15",
  "currency": "CNY"
},

"2025-08":{
  "card_number": "6211****1111",
  "cardholder_name": "张三",
  "bank": "中国银行",
  "bill_month": "2025-08",
  "total_amount": 15680.50,
  "minimum_payment": 1568.05,
  "payment_status": "unpaid",
  "due_date": "2025-10-15",
  "currency": "CNY"
},

"2025-07":{
  "card_number": "6211****1111",
  "cardholder_name": "张三",
  "bank": "中国银行",
  "bill_month": "2025-07",
  "total_amount": 13769.50,
  "minimum_payment": 1568.05,
  "payment_status": "unpaid",
  "due_date": "2025-10-15",
  "currency": "CNY"
},

"2025-06":{
  "card_number": "6211****1111",
  "cardholder_name": "张三",
  "bank": "中国银行",
  "bill_month": "2025-06",
  "total_amount": 34621.05,
  "minimum_payment": 1568.05,
  "payment_status": "unpaid",
  "due_date": "2025-10-15",
  "currency": "CNY"
}

}




def get_credit_card_monthly_bill(card_number: str, month: str) -> Dict:
    """
    查询指定信用卡的月度账单信息

    Args:
        card_number: 信用卡号（必填）
        month: 账单月份，格式YYYY-MM（必填）

    Returns:
        符合接口响应格式的账单信息字典，包含错误信息（如有）
    """

    # 查询信用卡数据
    card_info = MOCK_CREDIT_CARD_DATA.get(month)
    return card_info

from fastapi import FastAPI
import uvicorn

# FastAPI 应用
app = FastAPI(
    title="查询指定信用卡的月度账单信息",
    description="HTTP 服务",
    version="1.0.0",
)
from pydantic import BaseModel  # 用于定义请求参数模型（可选）

class QueryParams(BaseModel):
    card_number: str
    month: str
    # city: str = None  # 可选参数



from fastapi import FastAPI, Header, HTTPException
@app.get("/api/credit-card/monthly-bill")
async def call_tool(cardNumber: str, month: str,
                    X_App_Id: str = Header(..., description="用户令牌"),
                    X_App_Key: str = Header(..., description="应用密钥")
                    ):

    """
    需要通过 Header 传递鉴权参数的接口：
    - token: 用户令牌（必传，放在 Header 中）
    - app_key: 应用密钥（必传，放在 Header 中）
    """
    # 简单鉴权逻辑（实际场景需对接数据库或鉴权服务）
    valid_app_id = "your_app_id"
    valid_app_key = "your_app_key"

    if X_App_Id != valid_app_id or X_App_Key != valid_app_key:
        # 鉴权失败返回 401 错误
        raise HTTPException(
            status_code=401,
            detail="鉴权失败：app_id 或 app_key 无效"
        )
    """直接调用指定工具"""
    card_number = cardNumber
    month = month

    return get_credit_card_monthly_bill(card_number,month)


if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port=20006)