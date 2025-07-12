from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
import asyncio

import os
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 1. 从环境变量获取API密钥

from dotenv import load_dotenv
load_dotenv(dotenv_path="C:\\Users\\27296\\PycharmProjects\\fastApiProject\\.env")  # 加载.env文件
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("未找到DEEPSEEK_API_KEY环境变量，请先设置")

# 2. 创建支持流式输出的DeepSeek模型实例
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.7,
    streaming=True,  # 启用流式输出
    callbacks=[StreamingStdOutCallbackHandler()]  # 添加流式回调处理器
)

# 3. 创建带记忆的对话链
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    verbose=False
)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 明确指定前端地址
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)


class ChatRequest(BaseModel):
    message: List[Dict[str, str]]
@app.post("/message")
async def message_endpoint(prompt: ChatRequest):
    async def generate():
        response = conversation.predict(input=prompt.message)
        if response:
            yield f"data: {response}\n"
            await asyncio.sleep(0.02)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

