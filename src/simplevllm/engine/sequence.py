
import torch
from ..models.config import QWEN3_CONFIG

class SequenceState:
    def __init__(self, sid, prompt_ids, max_gen_len, QWEN3_CONFIG, device, matched_blocks=None):
        self.id, self.device, self.block_size = sid, device, QWEN3_CONFIG["block_size"]
        # Length tracking
        self.prompt_len = len(prompt_ids)
        self.prefix_len = len(matched_blocks) * self.block_size if matched_blocks else 0
        self.processed_len = self.prefix_len # How much has actually been computed
        self.num_tokens = self.prompt_len    # Total tokens (prompt + generated)
        
        self.max_total_len = self.prompt_len + max_gen_len
        self.tokens = torch.zeros(self.max_total_len, dtype=torch.long, device=device)
        self.tokens[:self.prompt_len] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        
        # Block Management
        max_blocks = (self.max_total_len + self.block_size - 1) // self.block_size
        self.block_table = torch.zeros(max_blocks, dtype=torch.int32, device=device)
        self.block_count = 0
        
        if matched_blocks:
            for b_id in matched_blocks:
                self.block_table[self.block_count] = b_id
                self.block_count += 1

        self.slot_mapping = torch.zeros(self.max_total_len, dtype=torch.long, device=device)

    @property
    def is_prefill(self):
        return self.processed_len < self.prompt_len

    def update_metadata(self, chunk_len):
        start = self.processed_len
        end = start + chunk_len
        idx = torch.arange(start, end, device=self.device)
        
        b_idx, o_idx = idx // self.block_size, idx % self.block_size
        # Map tokens to physical slots in the KV cache
        self.slot_mapping[idx] = self.block_table[b_idx].long() * self.block_size + o_idx
