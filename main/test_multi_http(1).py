import time

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def call_api_post(url,param):
    """调用HTTP接口的函数"""
    try:
        response = requests.post(url, json=param,timeout=60)

        return {
            "url": url,
            "status_code": response.status_code,
            "content": response.text # 只取部分内容
        }
    except Exception as e:
        return {
            "url": url,
            "error": str(e)
        }


import pandas as pd
import requests
import numpy as np
import json
def load_test_data():

    # 加载指定列的数据
    columns = ['segments', 'paper', 'id', 'category', 'question', 'content']
    df = pd.read_excel(file_path,usecols=columns)

    # # 转换为列表（每行是一个列表）
    # rows_list = df.values.tolist()
    # for i, row in enumerate(rows_list):
    #     print(f"第{i + 1}行：{row}")  # row[0]是第一列，row[1]是第二列...
    #
    # print(df.columns[:8])

    # 根据并发数生成批量数据
    def batch_generator(df,batch_size):
        """生成器：逐批返回DataFrame数据"""
        total_rows = len(df)
        for i in range(0, total_rows, batch_size):
            yield df.iloc[i:i + batch_size]

    # 使用生成器批量数据 并批量多并发请求

    all_result = []
    start = time.time()
    for i, batch_df in enumerate(batch_generator(df, batch_size=process_num), 1):
        print(f"第 {i} 批，包含 {len(batch_df)} 条数据：")
        print("-" * 50)
        # 转成字典列表
        # batch_data = batch_df.to_dict("records")
        batch_data = batch_df.fillna(np.nan).replace({np.nan: None}).to_dict("records")
        # 并发请求
        tmp_res_list = start_http_req(batch_data)
        # print(tmp_res_list)
        all_result.extend(tmp_res_list)

    # print(all_result)
    print(f"all cost {time.time()-start}")

    with open("sample_result.json", "w", encoding="utf-8") as f:
        json.dump(all_result, f, indent=2, ensure_ascii=False)

def start_http_req(params):
    tmp_result = []

    # 并发请求
    with ThreadPoolExecutor(max_workers=process_num) as executor:
        # 提交所有任务
        futures = {executor.submit(call_api_post, api_url,param): param for param in params}

        # 获取结果（按完成顺序返回）
        for future in as_completed(futures):
            url = futures[future]
            try:

                result = future.result()
                print(f"URL: {url}, 结果: {result}")
                # 解析步骤：
                # 1. 从result中提取content字段（JSON字符串）
                content_str = result['content']

                # 2. 将JSON字符串转换为Python字典
                content_dict = json.loads(content_str)

                # 3. 提取answer的值
                answer = content_dict.get('answer')
                print(f"结果: {answer}")
                tmp_result.append(answer)


            except Exception as e:
                print(f"URL: {url} 执行出错: {str(e)}")

                tmp_result.append(answer)

    return tmp_result


if __name__ == "__main__":

    file_path = "20251017 _1.xlsx"
    process_num = 1
    api_url = "http://127.0.0.1:10002/api/exam"
    load_test_data()

    # start_http_req()