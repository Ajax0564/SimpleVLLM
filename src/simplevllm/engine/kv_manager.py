import torch
from torch import nn
from collections import deque

class PagedKVManager(nn.Module):
    def __init__(self, cfg, max_blocks, device):
        super().__init__()
        self.block_size = cfg["block_size"]
        self.free_blocks = deque(range(max_blocks))
        self.evictable_blocks = deque() 
        
        self.hash_to_block_id = {}  
        self.block_id_to_hash = {}  
        self.block_ref_counts = {}   
        
        shape = (max_blocks, self.block_size, cfg["n_kv_groups"], cfg["head_dim"])
        self._k_cache_names = tuple(f"_k_cache_{i}" for i in range(cfg["n_layers"]))
        self._v_cache_names = tuple(f"_v_cache_{i}" for i in range(cfg["n_layers"]))

        for name in self._k_cache_names + self._v_cache_names:
            self.register_buffer(
                name,
                torch.empty(shape, device=device, dtype=cfg["dtype"]),
                persistent=False,
            )
    @property
    def k_cache(self):
        return [getattr(self, name) for name in self._k_cache_names]

    @property
    def v_cache(self):
        return [getattr(self, name) for name in self._v_cache_names]

    def get_prefix_blocks(self, token_ids):
        matched_blocks = []
        running_prefix = []
        for i in range(len(token_ids) // self.block_size):
            chunk = tuple(token_ids[i * self.block_size : (i + 1) * self.block_size])
            running_prefix.extend(chunk)
            p_hash = hash(tuple(running_prefix))
            
            if p_hash in self.hash_to_block_id:
                b_id = self.hash_to_block_id[p_hash]
                matched_blocks.append(b_id)
                if b_id in self.evictable_blocks:
                    self.evictable_blocks.remove(b_id)
                self.block_ref_counts[b_id] = self.block_ref_counts.get(b_id, 0) + 1
            else:
                break
        return matched_blocks

    def allocate(self, state, chunk_len):
        # Determine how many blocks we need based on the progress of the current chunk
        end_pos = state.processed_len + chunk_len
        needed_blocks = (end_pos + self.block_size - 1) // self.block_size
        
        while state.block_count < needed_blocks:
            if not self.free_blocks:
                if not self.evictable_blocks: raise RuntimeError("Out of KV Cache!")
                self._evict_lru()
            
            new_block = self.free_blocks.popleft()
            state.block_table[state.block_count] = new_block
            self.block_ref_counts[new_block] = 1
            state.block_count += 1

        # Register completed blocks in the prefix cache
        # We only register a block once its last token has been processed
        for b_idx in range(state.block_count):
            block_end_pos = (b_idx + 1) * self.block_size
            if end_pos >= block_end_pos:
                b_id = int(state.block_table[b_idx])
                if b_id not in self.block_id_to_hash:
                    self._register_block(state, b_idx, b_id)

    def _register_block(self, state, b_idx, b_id):
        prefix = tuple(state.tokens[:(b_idx + 1) * self.block_size].tolist())
        h = hash(prefix)
        self.hash_to_block_id[h] = b_id
        self.block_id_to_hash[b_id] = h

    def _evict_lru(self):
        evict_id = self.evictable_blocks.popleft()
        h = self.block_id_to_hash.pop(evict_id, None)
        if h: self.hash_to_block_id.pop(h, None)
        self.block_ref_counts.pop(evict_id, None)
        self.free_blocks.append(evict_id)

    def free(self, state):
        if getattr(state, "_freed", False):
            return

        state._freed = True
        for i in range(state.block_count):
            b_id = int(state.block_table[i])
            if b_id not in self.block_ref_counts:
                continue

            self.block_ref_counts[b_id] -= 1
            if self.block_ref_counts[b_id] <= 0:
                del self.block_ref_counts[b_id]
                if b_id in self.block_id_to_hash:
                    if b_id not in self.evictable_blocks:
                        self.evictable_blocks.append(b_id)
                else:
                    self.free_blocks.append(b_id)

    def stats(self):
        return {
            "free_blocks": len(self.free_blocks),
            "evictable_blocks": len(self.evictable_blocks),
            "used_blocks": len(self.block_ref_counts),
            "cached_prefixes": len(self.hash_to_block_id),
        }