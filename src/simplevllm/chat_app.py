import os
import threading
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .engine import ContinuousBatchEngine, PagedKVManager
from .models import Qwen3Config, get_qwen3_model, get_qwen3_tokenizer


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)


class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    history: list[dict[str, str]]


class ChatRuntime:
    def __init__(self, model=None, tokenizer=None, engine=None):
        if model is not None and tokenizer is not None and engine is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.engine = engine
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = get_qwen3_tokenizer()
            self.model = get_qwen3_model().to(self.device).eval()

            if self.device.type == "cuda" and os.getenv("SIMPLEVLLM_COMPILE") == "1":
                self.model = torch.compile(
                    self.model,
                    dynamic=True,
                    mode="reduce-overhead",
                    fullgraph=False,
                )

            max_blocks = int(os.getenv("SIMPLEVLLM_MAX_BLOCKS", "256"))
            self.kv_manager = PagedKVManager(
                Qwen3Config,
                max_blocks=max_blocks,
                device=self.device,
            )
            self.engine = ContinuousBatchEngine(
                self.model,
                self.kv_manager,
                Qwen3Config,
            )
        self.conversations: dict[str, list[dict[str, str]]] = {}
        self.lock = threading.Lock()

    def _encode_history(self, history: list[dict[str, str]]) -> list[int]:
        prompt_ids: list[int] = []
        for item in history:
            role = item["role"]
            content = item["content"]
            turn = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            prompt_ids.extend(self.tokenizer.encode(turn, chat_wrapped=False))

        assistant_header = "<|im_start|>assistant\n"
        prompt_ids.extend(self.tokenizer.encode(assistant_header, chat_wrapped=False))
        return prompt_ids

    def chat(self, request: ChatRequest) -> ChatResponse:
        conversation_id = request.conversation_id or uuid.uuid4().hex
        history = list(self.conversations.get(conversation_id, []))
        history.append({"role": "user", "content": request.message})
        prompt_ids = self._encode_history(history)

        if len(prompt_ids) + request.max_new_tokens > Qwen3Config["context_length"]:
            raise ValueError("Conversation is too long for the model context window")

        with self.lock:
            self.engine.reset()
            self.engine.add_sequence(prompt_ids, max_gen_len=request.max_new_tokens)
            finished: dict[int, list[int]] = {}
            while self.engine.waiting_room or self.engine.active:
                finished.update(self.engine.step())

        if not finished:
            raise RuntimeError("The engine did not produce a response")

        generated_ids = next(iter(finished.values()))[len(prompt_ids):]
        answer = self.tokenizer.decode(generated_ids)
        answer = answer.split("<|im_end|>", 1)[0].strip()
        history.append({"role": "assistant", "content": answer})
        self.conversations[conversation_id] = history
        return ChatResponse(
            conversation_id=conversation_id,
            message=answer,
            history=history,
        )


def main() -> None:
    import uvicorn

    uvicorn.run("simplevllm.chat_app:app", host="127.0.0.1", port=8000, reload=False)


def create_app(existing_runtime: ChatRuntime | None = None) -> FastAPI:
    application = FastAPI(title="SimpleVLLM Chat")
    application_runtime = existing_runtime

    @application.on_event("startup")
    def load_runtime() -> None:
        nonlocal application_runtime
        if application_runtime is None:
            application_runtime = ChatRuntime()

    @application.get("/")
    def index() -> FileResponse:
        return FileResponse(Path(__file__).with_name("static") / "index.html")

    @application.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        if application_runtime is None:
            raise HTTPException(status_code=503, detail="Model is still loading")
        try:
            return application_runtime.chat(request)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.delete("/api/chat/{conversation_id}", status_code=204)
    def clear_chat(conversation_id: str) -> None:
        if application_runtime is not None:
            application_runtime.conversations.pop(conversation_id, None)

    return application


app = create_app()
