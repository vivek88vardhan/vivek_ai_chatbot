from fastapi import FastAPI, Request
from llm_engine.ollama_client import query_ollama
from rag_engine.retriever import retrieve_docs
from local_api.api import get_local_data

app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data["message"]

    if "weather" in user_input:
        return {"response": get_local_data()}
    elif "search" in user_input:
        return {"response": retrieve_docs(user_input)}
    else:
        return {"response": query_ollama(user_input)}
