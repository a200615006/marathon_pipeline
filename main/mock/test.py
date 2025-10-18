import pandas as pd
import requests
import requests


def load_test_data():
    file_path = "./sample_A.xlsx"
    df = pd.read_excel(file_path)
    # print(df.columns)
    id_list = df["id"].tolist()
    category_list = df["category"].tolist()
    question_list = df["question"].tolist()
    content_list = df["content"].tolist()
    answer_list = df["answer"].tolist()
    label_list = df["label"].tolist()
    paper_list = df["paper"].tolist()
    segments_list = df["segments"].tolist()


    for i in range(len(id_list)):
        id = id_list[i]
        category = category_list[i]
        question = question_list[i]
        content = content_list[i]
        label = label_list[i]
        paper = paper_list[i]
        segments = segments_list[i]


        if category=="选择题":

            req_object = {
                "segments": segments,
                "paper": paper,
                "id": id,
                "category": category,
                "question": question,
                "content": content
            }
        else:
            req_object = {
                "segments": segments,
                "paper": paper,
                "id": id,
                "category": category,
                "question": question
            }

        # test_http_post(req_object)


def test_credit_card_month_bill():
    url = "http://127.0.0.1:20006/api/credit-card/monthly-bill"

    req_object = {
        "cardNumber": "1",
        "month": "2025-06"}

    header = {
        "X-App-Id":"your_app_id",
        "X-App-Key":"your_app_key"
    }
    response = requests.get(url, params=req_object,headers=header)
    # response = requests.get(url, params=req_object)

    # response = requests.get(url, params= params)
    print(response.text)


def test_exchange_rate():
    url = "http://127.0.0.1:20005/api/exchange-rate"

    req_object = {
        "fromCurrency": "JPY",
        "toCurrency": "KRW",
        "amount": "5000"

    }

    header = {
        "X-App-Id":"your_app_id",
        "X-App-Key":"your_app_key"
    }
    response = requests.get(url, params=req_object,headers=header)

    # response = requests.get(url, params=req_object)

    # response = requests.get(url, params= params)
    print(response.text)

def test_user_assets():
    url = "http://127.0.0.1:20007/api/user/assets"

    req_object = {
        "customerId": "110101199003072845",
        "assetType": "card"
    }

    header = {
        "X-App-Id":"your_app_id",
        "X-App-Key":"your_app_key"
    }
    response = requests.get(url, params=req_object,headers=header)

    # response = requests.get(url, params=req_object)

    # response = requests.get(url, params= params)
    print(response.text)

def test_utility_bill():
    url = "http://127.0.0.1:20008/api/utility-bill/monthly-bill"

    req_object = {
        "householdId": "BJ001234568",
        "month": "2025-08",
        "utilityType": "electricity"
    }

    header = {
        "X-App-Id":"your_app_id",
        "X-App-Key":"your_app_key"
    }
    response = requests.get(url, params=req_object,headers=header)

    # response = requests.get(url, params=req_object)

    # response = requests.get(url, params= params)
    print(response.text)


def test_payment_order():
    url = "http://127.0.0.1:20009/api/qr/create-payment-order"

    req_object = {
        "merchantId": "M001047",
        "orderId": "ORD002025147",
        "amount": "185"
    }

    header = {
        "X-App-Id":"your_app_id",
        "X-App-Key":"your_app_key"
    }
    response = requests.get(url, params=req_object,headers=header)

    # response = requests.get(url, params=req_object)

    # response = requests.get(url, params= params)
    print(response.text)

if __name__ == "__main__":
    # 启动FastAPI服务器
   # test_credit_card_month_bill()
     #test_exchange_rate()

     #test_utility_bill()
    #test_user_assets()
    test_payment_order()






