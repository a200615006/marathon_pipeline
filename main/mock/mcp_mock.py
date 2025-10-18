import re
from typing import Dict, Optional, List, Any
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import uvicorn

# 模拟数据 - 统一管理所有数据
MOCK_DATA: Dict[str, Dict] = {
    # 信用卡账单数据
    "621122221111_2025-09": {
        "card_number": "621122221111",
        "cardholder_name": "张三",
        "bank": "中国银行",
        "bill_month": "2025-09",
        "total_amount": 19345.50,
        "minimum_payment": 1568.05,
        "payment_status": "unpaid",
        "due_date": "2025-10-15",
        "currency": "CNY"
    },
    "621133331111_2025-09": {
        "card_number": "621133331111",
        "cardholder_name": "张三",
        "bank": "建设银行",
        "bill_month": "2025-09",
        "total_amount": 19580.30,
        "minimum_payment": 1958.05,
        "payment_status": "unpaid",
        "due_date": "2025-10-15",
        "currency": "CNY"
    },
    "6211****1111_2022-09": {
        "card_number": "6211****1111",
        "cardholder_name": "张三",
        "bank": "工商银行",
        "bill_month": "2022-09",
        "total_amount": 13769.50,
        "minimum_payment": 1568.05,
        "payment_status": "unpaid",
        "due_date": "2025-10-15",
        "currency": "CNY"
    },
    "621144441111_2025-09": {
        "card_number": "621144441111",
        "cardholder_name": "张三",
        "bank": "农夫银行",
        "bill_month": "2025-09",
        "total_amount": 34621.05,
        "minimum_payment": 1568.05,
        "payment_status": "unpaid",
        "due_date": "2025-10-15",
        "currency": "CNY"
    },

    # 汇率数据
    "USD_CNY": {
        "from_currency": "USD",
        "to_currency": "CNY",
        "rate": 7.09,
        "amount": 100,
        "converted_amount": 709.0,
        "timestamp": "2025-09-17T10:30:00"
    },
    "EUR_CNY": {
        "from_currency": "EUR",
        "to_currency": "CNY",
        "rate": 8.31,
        "amount": 100,
        "converted_amount": 831.0,
        "timestamp": "2025-09-17T10:30:00"
    },
    "JPY_CNY": {
        "from_currency": "JPY",
        "to_currency": "CNY",
        "rate": 4.73,
        "amount": 100,
        "converted_amount": 473.0,
        "timestamp": "2025-09-17T10:30:00"
    },
    "GBP_CNY": {
        "from_currency": "GBP",
        "to_currency": "CNY",
        "rate": 9.54,
        "amount": 100,
        "converted_amount": 954.0,
        "timestamp": "2025-09-17T10:30:00"
    },

    # 水电煤账单数据
    "BJ001234567_electricity_2025-09": {
        "household_id": "BJ001234567",
        "customer_name": "王二",
        "address": "北京市朝阳区建国门外大街1号",
        "utility_type": "electricity",
        "bill_month": "2025-09",
        "usage_amount": 174.8,
        "bill_amount": 266.8,
        "payment_status": "unpaid",
        "due_date": "2025-09-25",
        "currency": "CNY"
    },
    "BJ001234567_water_2025-09": {
        "household_id": "BJ001234567",
        "customer_name": "王二",
        "address": "北京市朝阳区建国门外大街1号",
        "utility_type": "water",
        "bill_month": "2025-09",
        "usage_amount": 253.1,
        "bill_amount": 132.3,
        "payment_status": "unpaid",
        "due_date": "2025-09-25",
        "currency": "CNY"
    },

    # 用户资产数据
    "110100002845_household": {
        "customer_id": "110100002845",
        "households": [
            {
                "household_id": "BJ001234567",
                "customer_name": "张元",
                "address": "北京市朝阳区建国门外大街1号1单元101室",
                "household_type": "住宅",
                "area": 43.5,
                "ownership_type": "私有",
                "registration_date": "2020-03-15"
            },
            {
                "household_id": "BJ001234567",
                "customer_name": "张元",
                "address": "北京市朝阳区建国门外大街1号1单元101室",
                "household_type": "住宅",
                "area": 50.5,
                "ownership_type": "私有",
                "registration_date": "2020-03-15"
            },
            {
                "household_id": "BJ001234568",
                "customer_name": "张元",
                "address": "北京市朝阳区商业街88号铺位203",
                "household_type": "商铺",
                "area": 65.2,
                "ownership_type": "共有",
                "registration_date": "2019-08-20"
            }
        ]
    },

    # 支付订单数据
    "M001047_ORD002025147": {
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

# 鉴权配置
VALID_APP_ID = "your_app_id"
VALID_APP_KEY = "your_app_key"


def authenticate_app(app_id: str, app_key: str) -> bool:
    """验证应用ID和密钥"""
    return app_id == VALID_APP_ID and app_key == VALID_APP_KEY


# FastAPI 应用
app = FastAPI(
    title="综合金融服务API",
    description="提供信用卡账单查询、汇率查询、水电煤账单查询、用户资产查询和支付订单创建等服务",
    version="1.0.0",
)


# 1. 信用卡账单查询服务
@app.get("/api/credit-card/monthly-bill")
async def get_credit_card_bill(
        cardNumber: str,
        month: str,
        X_App_Id: str = Header(..., description="用户令牌"),
        X_App_Key: str = Header(..., description="应用密钥")
) -> Dict:
    """
    查询指定信用卡的月度账单信息

    Args:
        cardNumber: 信用卡号
        month: 账单月份，格式YYYY-MM
        X_App_Id: 用户令牌
        X_App_Key: 应用密钥
    """
    if not authenticate_app(X_App_Id, X_App_Key):
        raise HTTPException(status_code=401, detail="鉴权失败：app_id 或 app_key 无效")

    # 构建查询键
    key = f"{cardNumber}_{month}"
    bill_info = MOCK_DATA.get(key)

    if not bill_info:
        raise HTTPException(status_code=404, detail="未找到指定的信用卡账单信息")

    return bill_info


# 2. 汇率查询服务
@app.get("/api/exchange-rate")
async def get_exchange_rate(
        fromCurrency: str,
        toCurrency: str,
        amount: float = 100.0,
        X_App_Id: str = Header(..., description="用户令牌"),
        X_App_Key: str = Header(..., description="应用密钥")
) -> Dict:
    """
    提供实时汇率查询和货币转换功能

    Args:
        fromCurrency: 源货币代码
        toCurrency: 目标货币代码
        amount: 转换金额，默认100
        X_App_Id: 用户令牌
        X_App_Key: 应用密钥
    """
    if not authenticate_app(X_App_Id, X_App_Key):
        raise HTTPException(status_code=401, detail="鉴权失败：app_id 或 app_key 无效")

    # 构建查询键
    from_cur = fromCurrency.upper()
    to_cur = toCurrency.upper()
    rate_key = f"{from_cur}_{to_cur}"

    rate_info = MOCK_DATA.get(rate_key)
    if not rate_info:
        raise HTTPException(status_code=404, detail="未找到指定的汇率信息")

    # 如果提供了不同的金额，重新计算转换金额
    if amount != 100.0:
        rate_info = rate_info.copy()
        original_rate = rate_info["rate"]
        rate_info["amount"] = amount
        rate_info["converted_amount"] = round(amount * original_rate, 2)

    return rate_info


# 3. 支付订单创建服务
@app.get("/api/qr/create-payment-order")
async def create_payment_order(
        merchantId: str,
        orderId: str,
        amount: Optional[float] = None,
        X_App_Id: str = Header(..., description="用户令牌"),
        X_App_Key: str = Header(..., description="应用密钥")
) -> Dict:
    """
    创建支付订单

    Args:
        merchantId: 商户号
        orderId: 订单号
        amount: 订单金额
        X_App_Id: 用户令牌
        X_App_Key: 应用密钥
    """
    if not authenticate_app(X_App_Id, X_App_Key):
        raise HTTPException(status_code=401, detail="鉴权失败：app_id 或 app_key 无效")

    key = f"{merchantId}_{orderId}"
    order_info = MOCK_DATA.get(key)

    if not order_info:
        raise HTTPException(status_code=404, detail="未找到指定的订单信息")

    # 如果提供了金额，更新订单金额
    if amount is not None:
        order_info = order_info.copy()
        order_info["amount"] = amount

    return order_info


# 4. 用户资产查询服务
@app.get("/api/user/assets")
async def get_user_assets(
        customerId: str,
        assetType: str = "household",
        X_App_Id: str = Header(..., description="用户令牌"),
        X_App_Key: str = Header(..., description="应用密钥")
) -> Dict:
    """
    根据用户ID查询用户名下的资产信息

    Args:
        customerId: 用户ID
        assetType: 资产类型，默认household
        X_App_Id: 用户令牌
        X_App_Key: 应用密钥
    """
    if not authenticate_app(X_App_Id, X_App_Key):
        raise HTTPException(status_code=401, detail="鉴权失败：app_id 或 app_key 无效")

    key = f"{customerId}_{assetType}"
    assets_info = MOCK_DATA.get(key)

    if not assets_info:
        raise HTTPException(status_code=404, detail="未找到指定的用户资产信息")

    return assets_info


# 5. 水电煤账单查询服务
@app.get("/api/utility-bill/monthly-bill")
async def get_utility_bill(
        householdId: str,
        month: str,
        utilityType: str = "electricity",
        X_App_Id: str = Header(..., description="用户令牌"),
        X_App_Key: str = Header(..., description="应用密钥")
) -> Dict:
    """
    查询指定户号的月度水电煤账单信息

    Args:
        householdId: 户号
        month: 账单月份，格式YYYY-MM
        utilityType: 账单类型，默认electricity
        X_App_Id: 用户令牌
        X_App_Key: 应用密钥
    """
    if not authenticate_app(X_App_Id, X_App_Key):
        raise HTTPException(status_code=401, detail="鉴权失败：app_id 或 app_key 无效")

    key = f"{householdId}_{utilityType}_{month}"
    bill_info = MOCK_DATA.get(key)

    if not bill_info:
        raise HTTPException(status_code=404, detail="未找到指定的水电煤账单信息")

    return bill_info


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=20006)