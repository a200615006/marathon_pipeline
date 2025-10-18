import re
import random
import string
from datetime import datetime, timedelta
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

"M001047_ORD002025147":{
  "payment_order_id": "PO_828EF3307FEC",
  "merchant_id": "M001047",
  "order_id": "ORD002025147",
  "amount": 185,
  "payment_status": "PENDING",
  "expire_time": "2025-09-19T16:05:24.123456",
  "timestamp": "2025-09-19T15:35:24.123456",
  "message": "Payment order created successfully"
}

}





from fastapi import FastAPI,Header,HTTPException
import uvicorn

# FastAPI 应用
app = FastAPI(
    title="提供实时汇率查询和货币转换功能服务",
    description="HTTP 服务",
    version="1.0.0",
)

@app.get("/api/qr/create-payment-order")
def create_payment_order(
        merchantId: str,
        orderId: str,
        amount: Optional[float] = None,
        X_App_Id: str = Header(..., description="用户令牌"),
        X_App_Key: str = Header(..., description="应用密钥")
) -> Dict:
    """
    创建支付订单工具函数

    Args:
        merchant_id: 商户号（必填）
        order_id: 订单号（必填）
        amount: 订单金额（可选，默认为0.0）

    Returns:
        符合接口响应格式的订单信息字典，包含错误信息（如有）
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
    merchant_id = merchantId
    order_id = orderId
    amount = amount
    key = merchant_id + "_" + order_id

    return MOCK_CREDIT_CARD_DATA.get(key)


if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port=20009)