import requests


def post():
    # url = "http://localhost:10002/query"
    url = "http://localhost:10002/api/exam"
    # data = {"query": "帮我计算计算下9+7*6/2 "}
    # data = {"query": "2025年9月18日起，民生银行信用卡中心将对哪项业务纳入信用卡资金受控金额？"}
    # data = {"query": "汇率转换：5000日元等于多少韩元"}
    # data = {"query": "查询户号BJ001234568在2025-08的电使用量为多少度"}
    # data = {"query": "请查询机构名称为建设银行且交易类型为REFUND的最近2笔交易的交易ID和金额，以交易时间倒序排序。请通过数据查询方式获取结果，最终仅返回结果值"}
    data = {"query": "计算7+8/2+5",
            "segments": "1",
            "paper": "exam",
            "id": 1,
            "category": "选择题",
            "question": "计算7+8/2+5",
            "content": "选项：\\nA) 6  \\nB) 16 \\nC) 23 \\nD) 15"
            }
    response = requests.post(url, json=data)
    print(response.text)


if __name__ == "__main__":
    post()
