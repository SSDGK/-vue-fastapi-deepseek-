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


from langchain.callbacks.base import AsyncCallbackHandler
class StreamingCallbackHandler(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        await self.queue.put(token)  # 将每个token放入队列


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
    # 创建异步队列
    queue = asyncio.Queue()

    # 创建支持流式的LLM实例
    streaming_llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        temperature=0.7,
        streaming=True,
        callbacks=[StreamingCallbackHandler(queue)]  # 使用自定义回调
    )

    # 创建带记忆的对话链
    conversation = ConversationChain(
        llm=streaming_llm,
        memory=ConversationBufferMemory(),
        verbose=False
    )

    # 加载历史对话
    for msg in prompt.message[:-1]:  # 最后一条是新消息
        if msg["role"] == "user":
            conversation.memory.save_context(
                {"input": msg["content"]},
                {"output": ""}
            )
        elif msg["role"] == "assistant":
            conversation.memory.save_context(
                {"input": ""},
                {"output": msg["content"]}
            )

    # 异步生成响应
    async def generate():
        # 在后台运行预测
        task = asyncio.create_task(
            conversation.apredict(input=prompt.message[-1]["content"])
        )

        # 流式返回token
        while True:
            token = await queue.get()
            if token is None:
                break
            yield f"{token}"
            await asyncio.sleep(0.01)  # 控制速率

        await task  # 确保任务完成

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )
