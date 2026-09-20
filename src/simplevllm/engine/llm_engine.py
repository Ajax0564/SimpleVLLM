import itertools
from collections import deque
import torch
from .sequence import SequenceState

class ContinuousBatchEngineNaive:
    def __init__(self, model, kv_mgr, cfg):
        self.model = model
        self.model.eval()
        self.kv_mgr = kv_mgr
        self.cfg = cfg
        self.active = {}
        self.id_gen = itertools.count()
        self.device = next(model.parameters()).device
        
        self.max_batch = cfg.get("max_batch_size", 4)
        self.chunk_size = cfg.get("chunk_size", 256) 
        self.waiting_room = deque()


    def add_sequence(self, prompt_ids, max_gen_len=128):
        if not prompt_ids:
            raise ValueError("prompt_ids cannot be empty")

        max_length = self.cfg["context_length"]
        if len(prompt_ids) + max_gen_len > max_length:
            raise ValueError(
                f"Prompt plus generation length exceeds context length {max_length}"
            )

        sid = next(self.id_gen)
        self.waiting_room.append({
            "sid": sid,
            "prompt_ids": list(prompt_ids),
            "max_gen_len": max_gen_len,
        })

    def reset(self):
        for state in list(self.active.values()):
            self.kv_mgr.free(state)

        self.active.clear()
        self.waiting_room.clear()
        self.id_gen = itertools.count()

    def stats(self):
        return {
            "active_sequences": len(self.active),
            "waiting_sequences": len(self.waiting_room),
            "kv_cache": self.kv_mgr.stats(),
        }

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
        if not self.active:
            return {}

        states = list(self.active.values())
        chunks = [
            min(self.chunk_size, s.prompt_len - s.processed_len)
            if s.is_prefill else 1
            for s in states
        ]
        starts = [s.processed_len for s in states]
        is_decoding = all(not s.is_prefill for s in states)

        for s, chunk_len in zip(states, chunks):
            self.kv_mgr.allocate(s, chunk_len)
            s.update_metadata(chunk_len)

        ids_to_cat, pos_to_cat, slots_to_cat = [], [], []
        cu_q, cu_k, seqlens = [0], [0], []
        q_acc, k_acc = 0, 0

        for s, start, chunk_len in zip(states, starts, chunks):
            end = start + chunk_len
            ids_to_cat.append(s.tokens[start:end].to(self.device))
            pos_to_cat.append(torch.arange(start, end, device=self.device))
            slots_to_cat.append(s.slot_mapping[start:end].to(self.device))

            q_acc += chunk_len
            k_acc += end
            cu_q.append(q_acc)
            cu_k.append(k_acc)
            seqlens.append(end)

        block_size = self.cfg["block_size"]
        max_num_blocks_per_seq = (max(seqlens) + block_size - 1) // block_size
        block_table = torch.zeros(
            (len(states), max_num_blocks_per_seq),
            dtype=torch.int32,
            device=self.device,
        )
        for row, state in enumerate(states):
            if state.block_count > 0:
                copy_count = min(state.block_count, max_num_blocks_per_seq)
                block_table[row, :copy_count] = state.block_table[:copy_count].to(
                    device=self.device,
                    dtype=torch.int32,
                )

        token_positions = torch.cat(pos_to_cat, dim=0)
        meta = {
            "is_decoding": is_decoding,
            "slot_mapping": torch.cat(slots_to_cat, dim=0),
            "cu_seqlens_q": torch.tensor(cu_q, dtype=torch.int32, device=self.device),
            "cu_seqlens_k": torch.tensor(cu_k, dtype=torch.int32, device=self.device),
            "max_seqlen_q": max(chunks),
            "max_seqlen_k": max(seqlens),
            "seqlens": torch.tensor(seqlens, dtype=torch.int32, device=self.device),
            "block_table": block_table,
            "block_size": block_size,
            "cos": self.model.cos_buf[token_positions].unsqueeze(1),
            "sin": self.model.sin_buf[token_positions].unsqueeze(1),
        }

        with torch.no_grad():
            logits = self.model(
                torch.cat(ids_to_cat, dim=0),
                self.kv_mgr.k_cache,
                self.kv_mgr.v_cache,
                meta,
            )

        finished = {}
        curr_offset = 0
        next_tokens = torch.argmax(logits, dim=-1)

        for state, chunk_len in zip(states, chunks):
            last_idx = curr_offset + chunk_len - 1

            if state.processed_len >= state.prompt_len:
                token_id = next_tokens[last_idx].item()
                should_continue = state.append_token(token_id)

                if not should_continue:
                    finished[state.id] = state.tokens[:state.num_tokens].tolist()
                    self.kv_mgr.free(state)
                    del self.active[state.id]

            curr_offset += chunk_len

        return finished

class ContinuousBatchEngine(ContinuousBatchEngineNaive):
    """ContinuousBatchEngine with reusable GPU staging buffers."""

    def __init__(self, model, kv_mgr, cfg):
        super().__init__(model, kv_mgr, cfg)

        self.max_tokens_per_step = self.max_batch * self.chunk_size
        self.max_blocks_per_seq = (
            self.cfg["context_length"] + self.cfg["block_size"] - 1
        ) // self.cfg["block_size"]

        self.gpu_input_ids = torch.empty(
            self.max_tokens_per_step,
            dtype=torch.long,
            device=self.device,
        )
        self.gpu_slot_mapping = torch.empty(
            self.max_tokens_per_step,
            dtype=torch.long,
            device=self.device,
        )
        self.gpu_positions = torch.empty(
            self.max_tokens_per_step,
            dtype=torch.long,
            device=self.device,
        )
        self.gpu_position_template = torch.arange(
            self.max_tokens_per_step,
            dtype=torch.long,
            device=self.device,
        )
        self.gpu_block_table = torch.empty(
            self.max_batch,
            self.max_blocks_per_seq,
            dtype=torch.int32,
            device=self.device,
        )
        self.gpu_block_table.zero_()
        self.gpu_sequence_block_table = torch.empty_like(self.gpu_block_table)
        self.gpu_sequence_block_table.zero_()
        self.gpu_sequence_rows = {}
        self.gpu_sequence_block_counts = {}
        self.free_gpu_rows = deque(range(self.max_batch))

        self.gpu_cu_q = torch.empty(
            self.max_batch + 1,
            dtype=torch.int32,
            device=self.device,
        )
        self.gpu_cu_k = torch.empty(
            self.max_batch + 1,
            dtype=torch.int32,
            device=self.device,
        )
        self.gpu_seqlens = torch.empty(
            self.max_batch,
            dtype=torch.int32,
            device=self.device,
        )
        self.gpu_selected_indices = torch.empty(
            self.max_batch,
            dtype=torch.long,
            device=self.device,
        )
        self.gpu_selected_tokens = torch.empty(
            self.max_batch,
            dtype=torch.long,
            device=self.device,
        )
        self.gpu_cos = torch.empty(
            self.max_tokens_per_step,
            self.model.cos_buf.shape[-1],
            dtype=self.model.cos_buf.dtype,
            device=self.device,
        )
        self.gpu_sin = torch.empty_like(self.gpu_cos)

    def _get_gpu_sequence_row(self, state):
        if state.id not in self.gpu_sequence_rows:
            if not self.free_gpu_rows:
                raise RuntimeError("No reusable GPU sequence rows available")
            row = self.free_gpu_rows.popleft()
            self.gpu_sequence_rows[state.id] = row
            self.gpu_sequence_block_counts[state.id] = 0
            self.gpu_sequence_block_table[row].zero_()
        return self.gpu_sequence_rows[state.id]

    def _release_gpu_sequence_row(self, state_id):
        row = self.gpu_sequence_rows.pop(state_id, None)
        self.gpu_sequence_block_counts.pop(state_id, None)
        if row is not None:
            self.gpu_sequence_block_table[row].zero_()
            self.free_gpu_rows.append(row)

    def reset(self):
        super().reset()
        self.gpu_sequence_rows.clear()
        self.gpu_sequence_block_counts.clear()
        self.free_gpu_rows = deque(range(self.max_batch))
        self.gpu_block_table.zero_()
        self.gpu_sequence_block_table.zero_()

    def step(self):
        self._try_schedule_waiting()
        if not self.active:
            return {}

        states = list(self.active.values())
        chunks = [
            min(self.chunk_size, s.prompt_len - s.processed_len)
            if s.is_prefill else 1
            for s in states
        ]
        starts = [s.processed_len for s in states]
        is_decoding = all(not s.is_prefill for s in states)
        non_blocking = self.device.type == "cuda"
        batch_size = len(states)

        for state, chunk_len in zip(states, chunks):
            self.kv_mgr.allocate(state, chunk_len)
            state.update_metadata(chunk_len)

        self.gpu_cu_q[0] = 0
        self.gpu_cu_k[0] = 0
        q_acc, k_acc = 0, 0
        token_offset = 0
        max_num_blocks_per_seq = 0
        max_seqlen_k = 0
        block_size = self.cfg["block_size"]

        for row, (state, start, chunk_len) in enumerate(
            zip(states, starts, chunks)
        ):
            end = start + chunk_len
            next_offset = token_offset + chunk_len

            self.gpu_input_ids[token_offset:next_offset].copy_(
                state.tokens[start:end],
                non_blocking=non_blocking,
            )
            self.gpu_slot_mapping[token_offset:next_offset].copy_(
                state.slot_mapping[start:end],
                non_blocking=non_blocking,
            )
            self.gpu_positions[token_offset:next_offset].copy_(
                self.gpu_position_template[:chunk_len]
            )
            self.gpu_positions[token_offset:next_offset].add_(start)
            self.gpu_cos[token_offset:next_offset].copy_(
                self.model.cos_buf[start:end],
                non_blocking=non_blocking,
            )
            self.gpu_sin[token_offset:next_offset].copy_(
                self.model.sin_buf[start:end],
                non_blocking=non_blocking,
            )

            q_acc += chunk_len
            k_acc += end
            max_seqlen_k = max(max_seqlen_k, end)
            self.gpu_cu_q[row + 1] = q_acc
            self.gpu_cu_k[row + 1] = k_acc
            self.gpu_seqlens[row] = end

            max_num_blocks_per_seq = max(
                max_num_blocks_per_seq,
                (end + block_size - 1) // block_size,
            )

            sequence_row = self._get_gpu_sequence_row(state)
            previous_count = self.gpu_sequence_block_counts[state.id]
            current_count = state.block_count
            if current_count > previous_count:
                self.gpu_sequence_block_table[
                    sequence_row,
                    previous_count:current_count,
                ].copy_(
                    state.block_table[previous_count:current_count],
                    non_blocking=non_blocking,
                )
                self.gpu_sequence_block_counts[state.id] = current_count

            token_offset = next_offset

        active_block_table = self.gpu_block_table[
            :batch_size,
            :max_num_blocks_per_seq,
        ]
        active_block_table.zero_()
        for row, state in enumerate(states):
            sequence_row = self.gpu_sequence_rows[state.id]
            active_block_table[row].copy_(
                self.gpu_sequence_block_table[
                    sequence_row,
                    :max_num_blocks_per_seq,
                ]
            )

        meta = {
            "is_decoding": is_decoding,
            "slot_mapping": self.gpu_slot_mapping[:token_offset],
            "cu_seqlens_q": self.gpu_cu_q[:batch_size + 1],
            "cu_seqlens_k": self.gpu_cu_k[:batch_size + 1],
            "max_seqlen_q": max(chunks),
            "max_seqlen_k": max_seqlen_k,
            "seqlens": self.gpu_seqlens[:batch_size],
            "block_table": active_block_table,
            "block_size": block_size,
            "cos": self.gpu_cos[:token_offset].unsqueeze(1),
            "sin": self.gpu_sin[:token_offset].unsqueeze(1),
        }

        with torch.inference_mode():
            logits = self.model(
                self.gpu_input_ids[:token_offset],
                self.kv_mgr.k_cache,
                self.kv_mgr.v_cache,
                meta,
            )

        finished = {}
        curr_offset = 0
        next_tokens = torch.argmax(logits, dim=-1)
        selected_count = 0
        selected_states = []

        for state, chunk_len in zip(states, chunks):
            last_idx = curr_offset + chunk_len - 1
            if state.processed_len >= state.prompt_len:
                self.gpu_selected_indices[selected_count] = last_idx
                selected_states.append(state)
                selected_count += 1
            curr_offset += chunk_len

        if selected_count:
            torch.gather(
                next_tokens,
                0,
                self.gpu_selected_indices[:selected_count],
                out=self.gpu_selected_tokens[:selected_count],
            )
            selected_tokens = self.gpu_selected_tokens[:selected_count].cpu().tolist()

            for state, token_id in zip(selected_states, selected_tokens):
                should_continue = state.append_token(token_id)
                if not should_continue:
                    finished[state.id] = state.tokens[:state.num_tokens].tolist()
                    self.kv_mgr.free(state)
                    self._release_gpu_sequence_row(state.id)
                    del self.active[state.id]

        return finished