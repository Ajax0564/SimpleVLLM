import torch
class SequenceState:
    def __init__(self, sid, prompt_ids, max_gen_len, cfg, device, matched_blocks=None):
        if not prompt_ids:
            raise ValueError("prompt_ids cannot be empty")
        if max_gen_len <= 0:
            raise ValueError("max_gen_len must be positive")
        self.id = sid
        self.device = device
        self.block_size = cfg["block_size"]
        self.eos_token_ids = set(cfg.get("eos_token_ids", [151643, 151645]))

        prompt_ids = list(prompt_ids)
        self.prompt_len = len(prompt_ids)
        matched_blocks = tuple(matched_blocks or ())
        self.prefix_len = len(matched_blocks) * self.block_size

        if self.prefix_len > self.prompt_len:
            raise ValueError("matched_blocks cannot extend beyond the prompt")

        self.max_total_len = self.prompt_len + max_gen_len
        self.processed_len = self.prefix_len
        self.num_tokens = self.prompt_len

        self.tokens = torch.empty(
            self.max_total_len,
            dtype=torch.long,
            device="cpu",
            pin_memory=True)
        self.tokens.zero_()
        self.tokens[: self.prompt_len] = torch.tensor(prompt_ids, dtype=torch.long, device="cpu")

        max_blocks = (self.max_total_len + self.block_size - 1) // self.block_size
        self.block_table = torch.empty(max_blocks,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True)
        self.block_table.zero_()
        self.block_count = 0

        if matched_blocks:
            self.block_table[: len(matched_blocks)] = torch.as_tensor(
                matched_blocks, dtype=self.block_table.dtype, device="cpu"
            )
            self.block_count = len(matched_blocks)

        self.slot_mapping = torch.empty(
        self.max_total_len,
        dtype=torch.long,
        device="cpu",
        pin_memory=True)
        self.slot_mapping.zero_()

    @property
    def is_prefill(self):
        return self.processed_len < self.prompt_len

    def append_token(self, token_id):
        if self.num_tokens >= self.max_total_len:
            return False

        self.tokens[self.num_tokens] = token_id
        self.num_tokens += 1
        return token_id not in self.eos_token_ids

    def update_metadata(self, chunk_len):
        start = self.processed_len
        end = start + chunk_len

        if chunk_len <= 0:
            raise ValueError("chunk_len must be positive")
        if end > self.num_tokens:
            raise ValueError("metadata range exceeds available tokens")
        if end > self.max_total_len:
            raise ValueError("metadata range exceeds sequence capacity")

        idx = torch.arange(start, end, dtype=torch.long, device="cpu")
        b_idx = idx // self.block_size
        o_idx = idx % self.block_size

        self.slot_mapping[start:end] = (
            self.block_table[b_idx].to(dtype=self.slot_mapping.dtype) * self.block_size + o_idx
        )
        self.processed_len = end