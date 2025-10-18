import re
from datetime import datetime
from typing import Dict, Optional, Tuple

# 支持的货币代码列表


MOCK_CREDIT_CARD_DATA: Dict[str, Dict] = {

"GBP_CNY":{
    "from_currency": "GBP",
    "to_currency": "CNY",
    "rate": 9.54,
    "amount": 100,
    "converted_amount": 954.0,
    "timestamp": "2025-09-17T10:30:00"
},

"USD_CNY":{
    "from_currency": "USD",
    "to_currency": "CNY",
    "rate": 7.09,
    "amount": 100,
    "converted_amount": 709.0,
    "timestamp": "2025-09-17T10:30:00"
},

"EUR_CNY":{
    "from_currency": "EUR",
    "to_currency": "CNY",
    "rate": 8.31,
    "amount": 100,
    "converted_amount": 831.0,
    "timestamp": "2025-09-17T10:30:00"
},

"JPY_KRW":{
    "from_currency": "JPY",
    "to_currency": "KRW",
    "rate": 9.09091,
    "amount": 5000,
    "converted_amount": 45454.55,
    "timestamp": "2025-09-17T10:30:00"
}

}


def exchange_currency(from_currency: str, to_currency: str, amount: float = 1.0) -> Dict:
    """
    货币转换工具函数

    Args:
        from_currency: 源货币代码
        to_currency: 目标货币代码
        amount: 转换金额，默认为1.0

    Returns:
        包含转换结果的字典，格式符合接口响应要求
    """

    # 获取汇率
    from_cur = from_currency.upper()
    to_cur = to_currency.upper()

    # 获取汇率
    rate_key = f"{from_cur}_{to_cur}"
    rate = MOCK_CREDIT_CARD_DATA.get(rate_key)
    return rate



from fastapi import FastAPI,Header,HTTPException
import uvicorn

# FastAPI 应用
app = FastAPI(
    title="提供实时汇率查询和货币转换功能服务",
    description="HTTP 服务",
    version="1.0.0",
)

@app.get("/api/exchange-rate")
async def call_tool(fromCurrency: str, toCurrency: str, amount: float = None,
                    X_App_Id: str = Header(..., description="用户令牌"),
                    X_App_Key: str = Header(..., description="应用密钥")
                    ):
    """直接调用指定工具"""

    # 简单鉴权逻辑（实际场景需对接数据库或鉴权服务）
    valid_app_id = "your_app_id"
    valid_app_key = "your_app_key"

    if X_App_Id != valid_app_id or X_App_Key != valid_app_key:
        # 鉴权失败返回 401 错误
        raise HTTPException(
            status_code=401,
            detail="鉴权失败：app_id 或 app_key 无效"
        )
    from_currency = fromCurrency
    to_currency = toCurrency
    amount = amount
    return exchange_currency(from_currency,to_currency,amount)



if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port=20005)
