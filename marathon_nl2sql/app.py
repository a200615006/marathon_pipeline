from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
import uvicorn

from config import TABLE_DESCRIPTIONS_CSV
from utils import get_table_details
from models import QueryRequest, QueryResponse
from chains import run_nl2sql_query

app = FastAPI(title="NL2SQL Service", description="自然语言转SQL查询服务")

table_details = get_table_details(TABLE_DESCRIPTIONS_CSV)

@app.post("/query", response_model=QueryResponse)
def process_query(request: QueryRequest):
    try:
        result = run_nl2sql_query(request.query)
        return QueryResponse(answer=result["answer"], sql_query=result["sql_query"])
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=18080)