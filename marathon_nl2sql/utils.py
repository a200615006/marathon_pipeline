# utils.py

import re
import pandas as pd


def clean_sql_query(text: str) -> str:
    """
    清理SQL查询：移除代码块语法、各种SQL标签、反引号、多余空白等，同时保留核心SQL查询。

    Args:
        text (str): 原始SQL查询文本，可能包含代码块、标签和反引号。

    Returns:
        str: 清理后的SQL查询。
    """
    # 步骤1: 移除代码块语法和SQL相关标签
    block_pattern = r"```(?:sql|SQL|SQLQuery|mysql|postgresql)?\s*(.*?)\s*```"
    text = re.sub(block_pattern, r"\1", text, flags=re.DOTALL)

    # 步骤2: 处理"SQLQuery:"前缀及其变体
    prefix_pattern = r"^(?:SQL\s*Query|SQLQuery|MySQL|PostgreSQL|SQL)\s*:\s*"
    text = re.sub(prefix_pattern, "", text, flags=re.IGNORECASE)

    # 步骤3: 如果有随机文本，提取第一个完整的SQL语句（以分号结束）
    sql_statement_pattern = r"(SELECT.*?;)"
    sql_match = re.search(sql_statement_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if sql_match:
        text = sql_match.group(1)

    # 步骤4: 移除标识符周围的反引号
    text = re.sub(r'`([^`]*)`', r'\1', text)

    # 步骤5: 规范化空白
    text = re.sub(r'\s+', ' ', text)

    # 步骤6: 为主要SQL关键字保留换行以保持可读性
    keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY',
                'LIMIT', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN',
                'OUTER JOIN', 'UNION', 'VALUES', 'INSERT', 'UPDATE', 'DELETE']
    pattern = '|'.join(r'\b{}\b'.format(k) for k in keywords)
    text = re.sub(f'({pattern})', r'\n\1', text, flags=re.IGNORECASE)

    # 步骤7: 最终清理
    text = text.strip()
    text = re.sub(r'\n\s*\n', '\n', text)

    return text


def get_table_details(csv_file: str) -> str:
    """
    从CSV文件读取数据库表描述，并格式化为字符串。

    Args:
        csv_file (str): CSV文件路径。

    Returns:
        str: 格式化的表描述字符串。

    Raises:
        ValueError: 如果文件未找到或读取出错。
    """
    try:
        table_df = pd.read_csv(csv_file)
        table_details = ""
        for index, row in table_df.iterrows():
            table_name = row['Table']
            table_desc = row['Description']
            fields = row.get('Fields', None)

            table_details += f"表名: {table_name}\n"
            table_details += f"描述: {table_desc}\n"

            if pd.notna(fields):
                field_list = fields.split(",")
                table_details += "字段描述:\n"
                for field in field_list:
                    table_details += f"  - {field.strip()}\n"
            else:
                table_details += "字段描述: 无可用描述\n"

            table_details += "\n"
        return table_details
    except FileNotFoundError:
        raise ValueError(f"CSV文件 '{csv_file}' 未找到。请确保文件存在。")
    except Exception as e:
        raise ValueError(f"读取CSV时出错: {str(e)}")