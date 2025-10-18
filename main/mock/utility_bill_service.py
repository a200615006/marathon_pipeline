import re
from typing import Dict, Optional

# 模拟信用卡账单数据（实际应用中可替换为数据库或API调用）
MOCK_CREDIT_CARD_DATA: Dict[str, Dict] = {

"BJ001234568_electricity_2025-08":{
    "household_id": "BJ001234568",
    "customer_name": "王二",
    "address": "北京市朝阳区建国门外大街1号",
    "utility_type": "electricity",
    "bill_month": "2025-08",
    "usage_amount": 22.9,
    "bill_amount": 266.8,
    "payment_status": "unpaid",
    "due_date": "2025-08-25",
    "currency": "CNY"
},
"BJ001234568_water":{
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
}

}


def get_utility_bill(
        household_id: str,
        month: str,
        utility_type: str = "electricity"
) -> Dict:
    """
    查询指定户号的月度水电煤账单信息

    Args:
        household_id: 户号（必填）
        month: 账单月份，格式YYYY-MM（必填）
        utility_type: 账单类型，可选值electricity/water/gas，默认electricity

    Returns:
        符合接口响应格式的账单信息字典，包含错误信息（如有）
    """
    key = household_id+ "_" + utility_type + "_" + month
    result = MOCK_CREDIT_CARD_DATA.get(key)


    return result


from fastapi import FastAPI,Header,HTTPException
import uvicorn

# FastAPI 应用
app = FastAPI(
    title="查询指定户号的月度水电煤账单信息服务",
    description="HTTP 服务",
    version="1.0.0",
)

@app.get("/api/utility-bill/monthly-bill")
async def call_tool(householdId: str,month: str,utilityType: str = None,
                    X_App_Id: str = Header(..., description="用户令牌"),
                    X_App_Key: str = Header(..., description="应用密钥")
                    ):
    # 简单鉴权逻辑（实际场景需对接数据库或鉴权服务）
    valid_app_id = "your_app_id"
    valid_app_key = "your_app_key"

    if X_App_Id != valid_app_id or X_App_Key != valid_app_key:
        # 鉴权失败返回 401 错误
        raise HTTPException(
            status_code=401,
            detail="鉴权失败：app_id 或 app_key 无效"
        )

    household_id = householdId
    month = month
    utility_type = utilityType
    return get_utility_bill(household_id,month,utility_type)



if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port=20008)





