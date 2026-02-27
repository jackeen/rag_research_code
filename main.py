from pydantic import BaseModel
from fastapi import FastAPI
from alpha import agent

app = FastAPI()


class ChatQueryModel(BaseModel):
    question: str


@app.get("/")
async def root():
    return {"message": "This is a chat system based on RAG"}


@app.post("/chat")
async def chat(chat_query: ChatQueryModel):
    res = agent.invoke_agent(chat_query.question)
    return {"response": res}
