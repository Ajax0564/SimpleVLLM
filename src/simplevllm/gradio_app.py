import os
import threading
import uuid

import gradio as gr
import torch

from .engine import ContinuousBatchEngine, PagedKVManager
from .models import Qwen3Config, get_qwen3_model, get_qwen3_tokenizer


class GradioRuntime:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = get_qwen3_tokenizer()
        self.model = get_qwen3_model().to(self.device).eval()

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
            turn = f"<|im_start|>{item['role']}\n{item['content']}<|im_end|>\n"
            prompt_ids.extend(self.tokenizer.encode(turn, chat_wrapped=False))

        prompt_ids.extend(
            self.tokenizer.encode("<|im_start|>assistant\n", chat_wrapped=False)
        )
        return prompt_ids

    def stream(self, conversation_id: str | None, message: str, max_new_tokens: int):
        conversation_id = conversation_id or uuid.uuid4().hex
        history = list(self.conversations.get(conversation_id, []))
        history.append({"role": "user", "content": message})
        prompt_ids = self._encode_history(history)

        if len(prompt_ids) + max_new_tokens > Qwen3Config["context_length"]:
            raise ValueError("Conversation is too long for the model context window")

        with self.lock:
            self.engine.reset()
            self.engine.add_sequence(prompt_ids, max_gen_len=max_new_tokens)
            generated_ids: list[int] | None = None
            answer = ""

            while self.engine.waiting_room or self.engine.active:
                for event in self.engine.step_single():
                    answer += self.tokenizer.decode([event["token_id"]])
                    answer = answer.split("<|im_end|>", 1)[0]
                    yield conversation_id, answer
                    if event["finished"]:
                        generated_ids = event["tokens"][len(prompt_ids):]

        if generated_ids is None:
            raise RuntimeError("The engine did not produce a response")

        final_answer = self.tokenizer.decode(generated_ids)
        final_answer = final_answer.split("<|im_end|>", 1)[0].strip()
        if not final_answer:
            final_answer = answer.strip()

        history.append({"role": "assistant", "content": final_answer})
        self.conversations[conversation_id] = history
        if final_answer != answer:
            yield conversation_id, final_answer


def create_demo(runtime: GradioRuntime | None = None) -> gr.Blocks:
    runtime = runtime or GradioRuntime()

    def respond(message, history, conversation_id, max_new_tokens):
        if not message or not message.strip():
            yield "", history, conversation_id
            return

        display_history = list(history or [])
        display_history.extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ""},
            ]
        )
        try:
            for conversation_id, answer in runtime.stream(
                conversation_id, message.strip(), int(max_new_tokens)
            ):
                display_history = [
                    *display_history[:-1],
                    {**display_history[-1], "content": answer},
                ]
                yield "", display_history, conversation_id
        except Exception as error:
            display_history = [
                *display_history[:-1],
                {**display_history[-1], "content": f"Error: {error}"},
            ]
            yield "", display_history, conversation_id

    def clear_chat():
        return [], None

    with gr.Blocks(title="SimpleVLLM Chat") as demo:
        gr.Markdown("# SimpleVLLM Chat")
        chatbot = gr.Chatbot(type="messages", height=600)
        conversation_id = gr.State(value=None)
        with gr.Row():
            message = gr.Textbox(
                placeholder="Message SimpleVLLM...",
                show_label=False,
                scale=8,
            )
            send = gr.Button("Send", variant="primary", scale=1)
        with gr.Row():
            max_new_tokens = gr.Slider(
                minimum=1,
                maximum=1024,
                value=256,
                step=1,
                label="Max new tokens",
            )
            clear = gr.Button("New chat")

        inputs = [message, chatbot, conversation_id, max_new_tokens]
        outputs = [message, chatbot, conversation_id]
        send.click(respond, inputs=inputs, outputs=outputs)
        message.submit(respond, inputs=inputs, outputs=outputs)
        clear.click(clear_chat, outputs=[chatbot, conversation_id])

    return demo


def main() -> None:
    create_demo().queue().launch()


if __name__ == "__main__":
    main()