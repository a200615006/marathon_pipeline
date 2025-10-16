# marathon_pipeline
2025马拉松总项目

## 安装
pip install -r requirements.txt

## 运行
需分别运行三个服务进程
marathon_nl2sql/run.sh   服务端口:18080
marathon_rag/run.sh  服务端口:28080
main/run.sh   服务端口:10000

## 使用示例

### nl2sql

使用POST请求到 /query，body: {"query": "销量最高的产品是什么"}
返回: {"answer": "结果答案", "sql_query": "生成的SQL"}\
例如:
curl -X POST http://localhost:18080/query \
     -H "Content-Type: application/json" \
     -d '{"query": "请查询机构名称为建设银行且交易类型为REFUND的最近2笔交易的交易ID和金额，以交易时间倒序排序。请通过数据查询方式获取结果，最终仅返回结果值"}'
