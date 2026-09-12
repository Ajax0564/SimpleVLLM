from __future__ import annotations

from collections.abc import Iterable

import torch

from .engine.kv_manager import PagedKVManager
from .engine.llm_engine import ContinuousBatchEngine
from .models.config import QWEN3_CONFIG
from .models.qwen3 import get_qwen3_model
from .models.tokenizer import get_tokenizer


class LLM:
    """Small stateful interface for submitting prompts to the batch engine."""

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        max_blocks: int = 128,
        max_batch_size: int | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = get_tokenizer()

        config = dict(QWEN3_CONFIG)
        if max_batch_size is not None:
            config["max_batch_size"] = max_batch_size

        self.model = get_qwen3_model(None).to(self.device)
        self.kv_manager = PagedKVManager(config, max_blocks=max_blocks, device=self.device)
        self.engine = ContinuousBatchEngine(self.model, self.kv_manager, config)
        self._completed: dict[int, list[int]] = {}

    def submit(self, prompt: str, max_new_tokens: int = 128) -> int:
        """Queue a prompt and return its request id."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        prompt_ids = self.tokenizer.encode(prompt)
        return self.engine.add_sequence(prompt_ids, max_gen_len=max_new_tokens)

    def _run_until(self, request_ids: set[int]) -> None:
        while request_ids - self._completed.keys():
            finished = self.engine.step()
            self._completed.update(finished)

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate one response, while preserving the running engine."""
        request_id = self.submit(prompt, max_new_tokens)
        self._run_until({request_id})
        token_ids = self._completed.pop(request_id)
        return self.tokenizer.decode(token_ids)

    def generate_batch(
        self, prompts: Iterable[str], max_new_tokens: int = 128
    ) -> list[str]:
        """Submit several prompts and return responses in input order."""
        request_ids = [self.submit(prompt, max_new_tokens) for prompt in prompts]
        self._run_until(set(request_ids))
        return [self.tokenizer.decode(self._completed.pop(request_id)) for request_id in request_ids]

    def chat(self, max_new_tokens: int = 128) -> None:
        """Read prompts until EOF or ``/exit`` and print each response."""
        print("Enter a prompt, or /exit to stop.")
        while True:
            try:
                prompt = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if prompt == "/exit":
                return
            if not prompt:
                continue
            print(self.generate(prompt, max_new_tokens=max_new_tokens))


def main() -> None:
    LLM().chat()


if __name__ == "__main__":
    main()
