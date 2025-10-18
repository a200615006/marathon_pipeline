import re
from typing import Dict, List, Optional

MOCK_CREDIT_CARD_DATA: Dict[str, Dict] = {

"110100002845":{
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
      "household_id": "BJ001234568",
      "customer_name": "张元",
      "address": "北京市朝阳区商业街88号铺位203",
      "household_type": "商铺",
      "area": 65.2,
      "ownership_type": "共有",
      "registration_date": "2019-08-20"
    }
  ]
}

}





def get_user_assets(customer_id: str, asset_type: Optional[str] = "card") -> Dict:
    """
    查询用户名下的资产信息（信用卡或房产）

    Args:
        customer_id: 用户ID（身份证号，必填）
        asset_type: 资产类型，可选值card/household，默认card

    Returns:
        符合接口响应格式的资产信息字典，包含错误信息（如有）
    """


    # 查询用户资产数据
    user_data = MOCK_CREDIT_CARD_DATA.get(customer_id)

    return user_data

from fastapi import FastAPI,Header,HTTPException
import uvicorn

# FastAPI 应用
app = FastAPI(
    title="根据用户ID查询用户名下的资产信息，包括信用卡资产和房产资产服务",
    description="HTTP 服务",
    version="1.0.0",
)

@app.get("/api/user/assets")
async def call_tool(customer_id: str, asset_type:str=None,
                    X_App_Id: str = Header(..., description="用户令牌"),
                    X_App_Key: str = Header(..., description="应用密钥")
                    ):

    """直接调用指定工具"""

    valid_app_id = "your_app_id"
    valid_app_key = "your_app_key"

    if X_App_Id != valid_app_id or X_App_Key != valid_app_key:
        # 鉴权失败返回 401 错误
        raise HTTPException(
            status_code=401,
            detail="鉴权失败：app_id 或 app_key 无效"
        )

    customer_id = customer_id
    asset_type = asset_type
    return get_user_assets(customer_id,asset_type)



if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port=20007)