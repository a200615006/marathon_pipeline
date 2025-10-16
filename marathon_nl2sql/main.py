from chains import create_nl2sql_chain
from utils import get_table_details
from config import TABLE_DESCRIPTIONS_CSV


def main():
    # 获取表描述
    table_details = get_table_details(TABLE_DESCRIPTIONS_CSV)

    # 创建链
    chain = create_nl2sql_chain(table_details)

    # 示例查询
    question = "销量最高的产品是什么"
    result = chain.invoke({"question": question, "table_details": table_details})
    print(f"问题: {question}")
    print(f"答案: {result}")

    # 另一个示例（生成查询但不执行）
    from chains import generate_query, clean_sql_query
    s_query = generate_query.invoke({"question": "统计一下谁买了`1910s Galleon`"})
    cleaned_query = clean_sql_query(s_query)
    print(f"生成的SQL查询: {cleaned_query}")


if __name__ == "__main__":
    main()