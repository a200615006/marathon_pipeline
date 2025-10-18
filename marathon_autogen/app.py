import asyncio
from fastapi import FastAPI, Body
from models import QueryRequest
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from agents import RAGAgent, NL2SQLAgent, MCPAgent, IntegratorAgent
from config import OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_API_KEY

app = FastAPI(title="AutoGen System API", description="基于 AutoGen 的工具调用系统")

# LLM 配置
model_client = OpenAIChatCompletionClient(
    model=MODEL_NAME,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE
)

@app.post("/process", response_model=dict)
async def process_query(request: QueryRequest = Body(...)):
    rag_agent = RAGAgent("rag_agent")
    nl2sql_agent = NL2SQLAgent("nl2sql_agent")
    mcp_agent = MCPAgent("mcp_agent")
    integrator_agent = IntegratorAgent("integrator_agent")

    # 配置 SelectorGroupChat（中性提示）
    selector_group_chat = SelectorGroupChat(
        agents=[rag_agent, nl2sql_agent, mcp_agent, integrator_agent],
        model_client=model_client,
        termination_condition=MaxMessageTermination(10),
        allow_repeated_speaker=True,
        selector_prompt=(
            "可用代理：{roles}\n其描述：{participants}\n"
            "基于查询 '{query}'，选择最合适的代理，仅返回代理名称。"
        ),
    )

    # 启动任务
    task = [TextMessage(content=request.query, source="user")]
    stream = selector_group_chat.run_stream(task=task)
    final_output = ""
    async for message in stream:
        if isinstance(message, TextMessage):
            final_output += message.content + "\n"

    return {"answer": final_output.strip()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=38080)