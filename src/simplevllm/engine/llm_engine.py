import itertools
from collections import deque
import torch
from .sequence import SequenceState

class ContinuousBatchEngine:
    def __init__(self, model, kv_mgr, cfg):
        self.model = model
        self.model.eval()
        self.kv_mgr = kv_mgr
        self.cfg = cfg
        self.active = {}
        self.id_gen = itertools.count()
        self.device = next(model.parameters()).device
        
        self.max_batch = cfg.get("max_batch_size", 4)
        self.chunk_size = cfg.get("chunk_size", 64) 
        self.waiting_room = deque()

    def add_sequence(self, prompt_ids, max_gen_len=128):
        sid = next(self.id_gen)
        self.waiting_room.append({
            'sid': sid, 'prompt_ids': prompt_ids, 'max_gen_len': max_gen_len
        })
        return sid

    def _try_schedule_waiting(self):
        while self.waiting_room and len(self.active) < self.max_batch:
            req = self.waiting_room[0]
            matched_blocks = self.kv_mgr.get_prefix_blocks(req['prompt_ids'])
            
            self.waiting_room.popleft()
            self.active[req['sid']] = SequenceState(
                req['sid'], req['prompt_ids'], req['max_gen_len'],
                self.cfg, self.device, matched_blocks=matched_blocks
            )

    def step(self):
        self._try_schedule_waiting()
        if not self.active: return {}

        states = list(self.active.values())
        chunks = [min(self.chunk_size, s.prompt_len - s.processed_len) if s.is_prefill else 1 for s in states]

        # Allocate and Map
        for s, c_len in zip(states, chunks):
            self.kv_mgr.allocate(s, c_len)
            s.update_metadata(c_len)

        # Prep metadata (Mixed Prefill/Decode)
        ids_to_cat, pos_to_cat, slots_to_cat = [], [], []
        cu_q, cu_k, seqlens = [0], [0], []
        q_acc, k_acc = 0, 0
        
        for s, c_len in zip(states, chunks):
            start, end = s.processed_len, s.processed_len + c_len
            ids_to_cat.append(s.tokens[start:end])
            pos_to_cat.append(torch.arange(start, end, device=self.device))
            slots_to_cat.append(s.slot_mapping[start:end])
            
            q_acc += c_len
            k_acc += end # Full KV history length
            cu_q.append(q_acc)
            cu_k.append(k_acc)
            seqlens.append(end)

        meta = {
            'is_decoding': all(not s.is_prefill for s in states),
            'slot_mapping': torch.cat(slots_to_cat),
            'cu_seqlens_q': torch.tensor(cu_q, dtype=torch.int32, device=self.device),
            'cu_seqlens_k': torch.tensor(cu_k, dtype=torch.int32, device=self.device),
            'max_seqlen_q': max(chunks),
            'max_seqlen_k': max(seqlens),
            'seqlens': torch.tensor(seqlens, dtype=torch.int32, device=self.device),
            'block_table': torch.stack([s.block_table for s in states]).to(torch.int32),
            'block_size': self.cfg["block_size"],
            'cos': self.model.cos_buf[torch.cat(pos_to_cat)].unsqueeze(1),
            'sin': self.model.sin_buf[torch.cat(pos_to_cat)].unsqueeze(1)
        }

        # Inference
        with torch.no_grad():
            logits = self.model(torch.cat(ids_to_cat), list(self.kv_mgr.k_cache), list(self.kv_mgr.v_cache), meta)

        # Post-Process
        finished = {}
        curr_offset = 0
        next_tokens = torch.argmax(logits, dim=-1)

        for s, c_len in zip(states, chunks):
            last_idx = curr_offset + c_len - 1
            s.processed_len += c_len
            
            # Sample only if we just finished the prompt or are in decoding phase
            if s.processed_len >= s.prompt_len:
                token_id = next_tokens[last_idx].item()
                s.tokens[s.num_tokens] = token_id
                s.num_tokens += 1
                
                if token_id in [151643, 151645] or s.num_tokens >= s.max_total_len:
                    finished[s.id] = s.tokens[:s.num_tokens].tolist()
                    self.kv_mgr.free(s)
                    del self.active[s.id]
            curr_offset += c_len

        return finished
