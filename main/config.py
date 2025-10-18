PROMPT_CHOICE = [{
            "role": "system",
            "content": """你是一个专业的选择题回答助手。请严格遵循以下要求：
                        1. 根据问题和选项分析并选择正确答案
                        2. 如果使用工具，请确保从问题中提取出完整的参数，未明确日期的年份按照当前年份2025
                        3. **返回答案时只返回选项字母（如A、B、C、D），不要包含任何其他文字、数字或符号**
                        4. **绝对不要返回选项内容或转换结果**
                        5. 输出格式必须为单个大写字母
                       """
        }]
PROMPT_QA = [{
    "role": "system",
    "content": """你是一个专业的MCP工具判断工具。请严格遵循以下要求：
                1、根据用户输入分析是否需要调用MCP工具，如需使用工具，请从用户输入中提取完整参数，严格遵守必选参数
                2、未明确日期的年份按照当前年份2025
                3、回复时只输出最终结果，不要任何计算过程、解释或额外文字
                4、如果是计算类问题，直接输出计算结果数字
                5、如果是工具调用，只输出工具返回的最终数据
                6、禁止添加任何说明性文字、公式或步骤
               """
}]
OPENAI_API_KEY = "sk-cb8dbe20f50d464a9707d93c30421ce5"
OPENAI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3-32b"
MCP_DIRECTORY="mcp_tools"

X_App_Id="your_app_id"
X_App_Key="your_app_key"
MAX_MCP_CALL=5
EXAM_PORT=10002
TEST_PORT=20001

MAIN_LOG_FILE="main.log"
NL2SQL_URL="http://localhost:18080/query"
RAG_URL="http://localhost:28080/query"