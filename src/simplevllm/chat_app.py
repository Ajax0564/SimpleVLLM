import os
import json
import threading
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
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
        response = None
        for event in self.stream_chat(request):
            if event["type"] == "done":
                response = ChatResponse.model_validate(event["response"])
                break
            if event["type"] == "error":
                raise ValueError(event["detail"])
        if response is None:
            raise RuntimeError("The engine did not produce a response")
        return response

    def stream_chat(self, request: ChatRequest):
        conversation_id = request.conversation_id or uuid.uuid4().hex
        history = list(self.conversations.get(conversation_id, []))
        history.append({"role": "user", "content": request.message})
        prompt_ids = self._encode_history(history)

        if len(prompt_ids) + request.max_new_tokens > Qwen3Config["context_length"]:
            raise ValueError("Conversation is too long for the model context window")

        with self.lock:
            self.engine.reset()
            self.engine.add_sequence(prompt_ids, max_gen_len=request.max_new_tokens)
            finished_tokens: list[int] | None = None
            while self.engine.waiting_room or self.engine.active:
                for event in self.engine.step_single():
                    text = self.tokenizer.decode([event["token_id"]])
                    if text:
                        yield {"type": "token", "text": text}
                    if event["finished"]:
                        finished_tokens = event["tokens"]

        if finished_tokens is None:
            raise RuntimeError("The engine did not produce a response")

        generated_ids = finished_tokens[len(prompt_ids):]
        answer = self.tokenizer.decode(generated_ids)
        answer = answer.split("<|im_end|>", 1)[0].strip()
        history.append({"role": "assistant", "content": answer})
        self.conversations[conversation_id] = history
        response = ChatResponse(
            conversation_id=conversation_id,
            message=answer,
            history=history,
        )
        yield {"type": "done", "response": response.model_dump()}


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
    def index() -> HTMLResponse:
        page = (Path(__file__).with_name("static") / "index.html").read_text()
        stream_adapter = """
<script>
(() => {
    const originalFetch = window.fetch;
    window.fetch = async (input, init) => {
        const url = typeof input === 'string' ? input : input.url;
        if (!url.endsWith('/api/chat') || !init || init.method !== 'POST') {
            return originalFetch(input, init);
        }
        const streamInit = {...init, body: init.body};
        const response = await originalFetch('/api/chat/stream', streamInit);
        if (!response.ok || !response.body) return response;
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalResponse = null;
        let assistantContent = null;
        const content = () => {
            if (assistantContent) return assistantContent;
            const wrapper = document.createElement('div');
            wrapper.className = 'message assistant';
            const label = document.createElement('div');
            label.className = 'label';
            label.textContent = 'Assistant';
            assistantContent = document.createElement('div');
            wrapper.append(label, assistantContent);
            document.querySelector('#messages').append(wrapper);
            return assistantContent;
        };
        const process = (line) => {
            if (!line.startsWith('data: ')) return;
            const event = JSON.parse(line.slice(6));
            if (event.type === 'token') {
                content().textContent += event.text;
                window.scrollTo(0, document.body.scrollHeight);
            } else if (event.type === 'done') {
                finalResponse = event.response;
            } else if (event.type === 'error') {
                throw new Error(event.detail);
            }
        };
        while (true) {
            const {value, done} = await reader.read();
            buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
            const lines = buffer.split('\n');
            buffer = lines.pop();
            for (const line of lines) process(line.trim());
            if (done) break;
        }
        if (buffer.trim()) process(buffer.trim());
        return new Response(JSON.stringify(finalResponse), {
            status: 200,
            headers: {'Content-Type': 'application/json'}
        });
    };
})();
</script>
"""
        return HTMLResponse(page.replace("</body>", stream_adapter + "</body>"))

    @application.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        if application_runtime is None:
            raise HTTPException(status_code=503, detail="Model is still loading")
        try:
            return application_runtime.chat(request)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.post("/api/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        if application_runtime is None:
            raise HTTPException(status_code=503, detail="Model is still loading")

        def events():
            try:
                for event in application_runtime.stream_chat(request):
                    yield f"data: {json.dumps(event)}\n\n"
            except ValueError as error:
                yield f"data: {json.dumps({'type': 'error', 'detail': str(error)})}\n\n"
            except Exception as error:
                yield f"data: {json.dumps({'type': 'error', 'detail': str(error)})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @application.delete("/api/chat/{conversation_id}", status_code=204)
    def clear_chat(conversation_id: str) -> None:
        if application_runtime is not None:
            application_runtime.conversations.pop(conversation_id, None)

    return application


app = create_app()
