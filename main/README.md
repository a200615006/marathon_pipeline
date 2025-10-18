# 1 进程总入口在 main/main.py 10000端口
# 2 发起请求运行 copy一份到test_post.py， 20001端口 

# 3 修改配置文件 config.py
- 考试端口号 10000
- 暴露测试 port 20001
- OPENAI_API_KEY
- OPENAI_API_BASE
- MODEL_NAME

- X_App_Id
- X_App_Key

- EXAM_PORT
- TEST_PORT

- NL2SQL_URL
- RAG_URL

# 4 修改 mcp_tools的ip地址

## 附测试阶段
### mock及测试
- mcp_mock服务在mock/mcp_mock.py, test.py测试mock
- 运行批量测试在main/test_multi_http.py，测试exam