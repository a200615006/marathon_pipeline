# NL2SQL FastAPI 服务

这是一个将自然语言查询转换为SQL查询的Web服务，使用LangChain和FastAPI。

## 安装
pip install -r requirements.txt

## 运行
python app.py
# 服务启动后，访问 http://localhost:8000/docs 查看API文档

## 使用示例
使用POST请求到 /query，body: {"query": "销量最高的产品是什么"}
返回: {"answer": "结果答案", "sql_query": "生成的SQL"}\
例如:
curl -X POST http://localhost:18080/query \
     -H "Content-Type: application/json" \
     -d '{"query": "销量最高的产品是什么"}'