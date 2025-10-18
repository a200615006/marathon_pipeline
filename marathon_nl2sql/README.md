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
     -d '{"query": "查询一下银联食堂10月份交易记录总和。请返回数值"}'

curl -X POST http://localhost:18080/query \
     -H "Content-Type: application/json" \
     -d '{"query": "张杰所在的商户8月份有几笔交易。请返回数值"}'

curl -X POST http://localhost:18080/query \
     -H "Content-Type: application/json" \
     -d '{"query": "肯德基7月份超过100的交易有几笔。请返回数值"}'

curl -X POST http://localhost:18080/query \
     -H "Content-Type: application/json" \
     -d '{"query": "查询机构名称为石河子银行且交易类型为REFUND的最近2笔交易的交易ID和金额，以交易时间倒序排序"}'


mysql -u root -p marathon < /root/marathon/marathon_pipeline/marathon_nl2sql/create_tables.sql
mysql -u root -p marathon < /root/marathon/marathon_pipeline/marathon_nl2sql/data.sql

     
