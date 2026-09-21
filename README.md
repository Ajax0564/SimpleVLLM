# SimpleVLLM

SimpleVLLM is a compact, learning-oriented implementation of a paged-attention inference engine inspired by vLLM-style scheduling and KV-cache reuse. It is designed to be easy to inspect and reason about while still capturing the core ideas behind modern LLM serving systems: sequence tracking, block-based KV cache management, and continuous batching.

This project is not meant to be a drop-in production replacement for full vLLM. Instead, it is a readable reference implementation for understanding how a minimal serving stack can:

- keep per-sequence state across prompt and generation phases,
- reuse KV blocks instead of recomputing full prefixes,
- schedule new requests while others are decoding,
- run batched inference efficiently on GPU.

## Quick local Gradio chat UI

Install the project and launch the ChatGPT-style Gradio interface with:

```bash
uv sync
uv run simplevllm-gradio
```

The first startup downloads the Qwen3 model and tokenizer, so it can take a while and requires a compatible CUDA environment for the current FlashAttention dependency. To enable model compilation on CUDA, set `SIMPLEVLLM_COMPILE=1` before starting.

The assistant message is updated as each token is generated. Use the `Max new tokens` slider to control the response length, or `New chat` to clear the conversation.

---

## 1. Why this architecture exists

The core challenge in serving LLMs is not only model execution, but memory management under variable-length requests.

Large language models process tokens step by step and keep Key/Value activations for every layer. These activations are expensive to store and are reused across decoding steps. If each request keeps a full contiguous memory block for all its tokens, memory becomes fragmented and requests with different lengths are hard to schedule efficiently.

The vLLM-like approach solves this by:

- splitting the KV cache into fixed-size blocks,
- assigning each sequence a block table describing which blocks belong to it,
- reusing prefix blocks when prompts overlap,
- batching active requests and sending only the needed token slices to the model.

This makes throughput higher and memory more predictable.

---

## 2. High-level runtime flow

The runtime roughly follows this sequence:

1. A request arrives with prompt token IDs and a generation budget.
2. It is placed into a waiting room.
3. The scheduler picks waiting sequences into the active set, respecting a batch limit.
4. For each active sequence, we determine how much work needs to happen this step:
   - prefill if the prompt is not fully processed,
   - decode if the sequence is already in generation mode.
5. The KV manager allocates or reuses blocks for the current token window.
6. The engine builds attention metadata such as sequence lengths, cumulative lengths, slot mappings, and block table.
7. The model is called on the batched token slice.
8. The next token is sampled and appended to the sequence state.
9. If generation ends or a sequence is finished, it is removed from the active set and its blocks are released.

This pattern is the heart of a continuous batching engine.

---

## 3. SequenceState: the per-sequence lifecycle manager

The `SequenceState` object is the main source of truth for one request while it is active.

It tracks:

- the sequence identifier,
- original prompt tokens,
- generation length limit,
- current processed position,
- token storage for prompt + generated output,
- block table entries for its KV blocks,
- slot mapping for each token position,
- whether the sequence is in prefill or decode mode.

### 3.1 Constructor responsibilities

When a sequence is created, we do the following:

- validate that the prompt is not empty,
- ensure the request fits inside `context_length`,
- set up `prompt_len`, `max_total_len`, and `processed_len`,
- allocate CPU buffers for the sequence token list and slot mapping,
- create a block table with room for future blocks,
- optionally attach already reused prefix blocks that match the beginning of the prompt.

The most important state fields are:

- `id`: unique sequence identifier
- `prompt_len`: length of the original prompt
- `processed_len`: how many tokens have already been consumed
- `num_tokens`: total number of tokens stored so far, including prompt + generated tokens
- `max_total_len`: total token capacity for the sequence
- `block_table`: list of KV cache block IDs assigned to this sequence
- `block_count`: number of blocks currently in use
- `slot_mapping`: physical slot index inside the KV cache for each token position
- `tokens`: full token history for the sequence

### 3.2 Prefill vs decode mode

`SequenceState.is_prefill` becomes true while the sequence has not yet processed its full prompt:

- if `processed_len < prompt_len`, the sequence still needs prefill work,
- after that, it transitions into decoding mode and generates one token at a time.

This distinction is critical because the batch step handles prompt processing and generation differently.

### 3.3 Token appending behavior

The method `append_token(token_id)` is the generation step for a sequence.

It adds the newly generated token into the sequence buffer and increments `num_tokens`, and returns whether generation should continue.

The continuation rule is usually based on whether the token is an EOS token:

- if the token is not an EOS token, the sequence continues,
- if it is EOS, generation ends and the sequence can be cleaned up.

This keeps generation logic simple and ensures the sequence terminates correctly.

### 3.4 Metadata update for attention

`update_metadata(chunk_len)` is the bridge between sequence-level bookkeeping and attention kernel metadata.

It does the following:

- determines the token range `[start, end)` for the current chunk,
- computes token indices for each position inside the chunk,
- maps those token positions to corresponding block indices and offsets,
- writes the correct `slot_mapping[start:end]` values,
- advances `processed_len` to the new end.

This is exactly what makes block-based attention possible: each token position knows which physical KV slot it should read from.

### 3.5 Why SequenceState matters

Without this object, the system would have no reliable way to:

- know how many tokens belong to each request,
- decide whether a request is under prefill or generation,
- know which cache blocks belong to each sequence,
- translate global token positions into per-block slot IDs.

In short, `SequenceState` is the state machine of the engine.

Assume:

- prompt tokens: `[101, 102, 103, 104, 105]`
- maximum generated tokens: `4`
- context length: `16`
- block size: `4`

Immediately after creating the request, the sequence state may look like this:

| Field | Value |
|---|---|
| `id` | `7` |
| `prompt_len` | `5` |
| `max_total_len` | `9` |
| `processed_len` | `0` |
| `num_tokens` | `5` |
| `tokens` | `[101, 102, 103, 104, 105]` |
| `is_prefill` | `true` |
| `block_table` | `[]` or reused prefix blocks |
| `block_count` | `0` |
| `slot_mapping` | empty or uninitialized |

The sequence has five prompt tokens, but none have been processed by the model yet.

If the prefill chunk size is `4`, the first step processes:

```text
[101, 102, 103, 104]
```

The sequence then has:

```text
processed_len = 4
num_tokens    = 5
is_prefill    = true
```

The next prefill step processes token `105`. After that:

```text
processed_len = 5
num_tokens    = 5
is_prefill    = false
```

The prompt is now fully present in the KV cache, so the sequence enters decode mode.

Assume the model generates the tokens `201`, `202`, and `203` during three decode steps. The state after those steps may look like:

| Field | Value |
|---|---|
| `id` | `7` |
| `prompt_len` | `5` |
| `max_total_len` | `9` |
| `processed_len` | `8` |
| `num_tokens` | `8` |
| `tokens` | `[101, 102, 103, 104, 105, 201, 202, 203]` |
| `is_prefill` | `false` |
| `block_table` | `[12, 19]` |
| `block_count` | `2` |

With a block size of `4`:

```text
token positions 0..3 -> block 12
token positions 4..7 -> block 19
```

The physical block IDs are allocated by the KV manager and are not required to be consecutive.

If token `204` is generated next, the state becomes:

```text
tokens       = [101, 102, 103, 104, 105, 201, 202, 203, 204]
num_tokens   = 9
processed_len = 9
```

A third block is now required because token position `8` belongs to the next block:

```text
token positions 8..11 -> block 23
block_table = [12, 19, 23]
```

The sequence stops early if a generated token is the EOS token. Otherwise, it continues until `max_total_len` is reached.


---

## 4. PagedKVManager: block allocator and prefix reuse logic

The `PagedKVManager` is responsible for memory management of the KV cache.

It acts like a block pool with a small prefix cache and reference counting.

### 4.1 What the manager owns

The manager maintains a fixed number of KV blocks for the entire runtime. Each block has capacity:

- block size = `cfg["block_size"]`
- number of KV heads = `cfg["n_kv_groups"]`
- head dimension = `cfg["head_dim"]`
- layer count = `cfg["n_layers"]`

For each layer, it creates a K and V cache buffer.

The manager exposes:

- `k_cache`: list of K caches per layer,
- `v_cache`: list of V caches per layer.

This allows the model to read/write the appropriate KV memory during attention.

### 4.2 Free blocks and evictable blocks

The manager uses:

- `free_blocks`: blocks that are completely available,
- `evictable_blocks`: blocks that are still in the eviction candidate set,
- `block_ref_counts`: how many sequences currently reference each block,
- `hash_to_block_id`: mapping from a prefix hash to a block id,
- `block_id_to_hash`: reverse map used during eviction.

This is a simplified block cache meant to mimic vLLM-style prefix reuse.

### 4.3 Prefix matching

`get_prefix_blocks(token_ids)` looks for reused prefix blocks by scanning token chunks in blocks of size `block_size`.

It builds a running prefix, hashes it, and checks whether that prefix already exists in `hash_to_block_id`. If it does, it reuses the cached block and increments its reference count.

This gives a significant gain when many prompts share the same prefix, such as system prompts or common instruction templates.

### 4.4 Allocation flow

When a sequence advances and more tokens need to be processed, `allocate(state, chunk_len)` determines how many blocks are required.

If the sequence needs more capacity than it currently has:

- it takes free blocks if available,
- otherwise it tries to evict an LRU-like candidate from `evictable_blocks`,
- it appends the new block to the sequence block table and updates its reference count.

Once a block is fully covered by the processed region, it can be registered in the prefix cache.

### 4.5 Registration and eviction

Block registration is a prefix optimization: a full block is remembered by hashing the prefix that produced it, making future prefix reuse cheaper.

Eviction is selective: if a block has no active references and may be reused later, it is moved to the evictable pool. This lets the scheduler keep available blocks while protecting common prefixes and reducing churn.

### 4.6 Why the KV manager matters

The KV manager is the memory backbone of the engine.

It ensures:

- requests do not overwrite each other's KV values,
- blocks are reused when possible,
- running sequences have enough cache space,
- finished sequences release memory and allow new ones to fill the pool.

This is what makes the serving stack efficient enough to behave like a vLLM-inspired runtime.

---

## 5. ContinuousBatchEngine: the scheduling core

`ContinuousBatchEngine` is the scheduler and execution engine that turns sequence state and KV memory into batched model calls.

It keeps two things:

- a `waiting_room` of not-yet-scheduled requests,
- an `active` dictionary of requests currently running.

### 5.1 Request admission

Requests enter through `add_sequence(prompt_ids, max_gen_len)`. This checks:

- prompt cannot be empty,
- request must not exceed model context length,
- a unique sequence ID is assigned,
- the request is appended to the waiting room.

### 5.2 Scheduling loop

The method `_try_schedule_waiting()` pulls waiting requests into `active` until the batch capacity is reached.

For each request, it calls `kv_mgr.get_prefix_blocks(...)` to recover any matching prefix blocks and constructs a `SequenceState` with that prefix metadata.

This is the moment when a new request becomes part of live generation.

### 5.3 Step execution pattern

`step()` is the main update loop.

Inside it:

1. schedule any waiting requests,
2. exit immediately if no active sequences remain,
3. compute a per-sequence chunk size:
   - prefill: use up to `chunk_size` tokens,
   - decode: one token at a time,
4. allocate KV blocks for each state using the manager,
5. update sequence metadata for the current chunk,
6. build model input tensors and attention metadata,
7. call the model with batched tokens,
8. sample the next token,
9. append generated tokens to sequences,
10. free finished sequences and remove them from `active`.

This pattern is what continuous batching means: a sequence can be in prefill or decode, and the engine keeps moving work forward efficiently.

### 5.4 Attention metadata built for the model

The model call receives a dictionary named `meta` that contains everything needed for attention.

Important fields include:

- `is_decoding`: whether all active sequences are in decode mode,
- `slot_mapping`: physical KV slots per token,
- `cu_seqlens_q`: cumulative sequence lengths for query tokens,
- `cu_seqlens_k`: cumulative sequence lengths for key/value tokens,
- `max_seqlen_q`: largest query length in the batch,
- `max_seqlen_k`: largest key/value sequence length in the batch,
- `seqlens`: per-sequence lengths,
- `block_table`: row-wise mapping of sequence to physical KV blocks,
- `block_size`: size of each block,
- `cos` and `sin`: RoPE or positional vectors for each token position.

This metadata is not optional: it tells the model how to read the correct KV values and how to align tokens belonging to different sequences in a packed batch.

### 5.5 Why `block_table` shape matters

A very important invariant is that the block table should be shaped like:

- `(batch_size, max_num_blocks_per_seq)`

This is a dense, padded matrix where each row corresponds to one sequence and each entry is a physical block index.

The engine uses this structure to tell the attention kernel which blocks belong to which sequence and where the sequence data can be found in the KV cache.

If the table is mis-shaped or incorrectly padded, attention will read the wrong memory and the runtime will fail with shape mismatch errors.

### 5.6 Reusable GPU staging buffers

The optimized `ContinuousBatchEngine` adds GPU staging buffers so that the runtime avoids re-allocating large tensors every step.

It creates preallocated GPU memory for:

- input token IDs,
- slot mappings,
- positions,
- cumulative lengths,
- block tables,
- cosine and sine buffers,
- chosen generation indices.

This is important for throughput because repeated `torch.empty(...)` and Python-side copies can become a bottleneck in small or medium batch workloads.

The engine keeps this optimization while preserving the same logical behavior as the naive version.

A block table translates logical token positions into physical KV-cache blocks.

Assume:

- block size: `4`
- sequence A tokens: `8`
- sequence B tokens: `5`
- sequence C tokens: `3`

The KV manager could assign these physical blocks:

```text
Sequence A:
  positions 0..3 -> block 12
  positions 4..7 -> block 19

Sequence B:
  positions 0..3 -> block 4
  positions 4..7 -> block 27

Sequence C:
  positions 0..2 -> block 8
```

The engine must create a rectangular, padded table for the batch. The maximum number of blocks required by any sequence is `2`, so the table is:

```text
block_table =
[
    [12, 19],   # sequence A
    [ 4, 27],   # sequence B
    [ 8, -1],   # sequence C; -1 is padding
]
```

The rows correspond to batch entries. The columns correspond to logical block positions within each sequence.

For example, sequence A's token position is mapped as follows:

```text
position 0 -> block 12, offset 0
position 1 -> block 12, offset 1
position 2 -> block 12, offset 2
position 3 -> block 12, offset 3

position 4 -> block 19, offset 0
position 5 -> block 19, offset 1
position 6 -> block 19, offset 2
position 7 -> block 19, offset 3
```

The physical KV slot is computed conceptually as:

```text
slot = block_id * block_size + offset
```

Therefore, with a block size of `4`, sequence A maps to:

```text
position 0 -> 12 * 4 + 0 = 48
position 1 -> 12 * 4 + 1 = 49
position 2 -> 12 * 4 + 2 = 50
position 3 -> 12 * 4 + 3 = 51

position 4 -> 19 * 4 + 0 = 76
position 5 -> 19 * 4 + 1 = 77
position 6 -> 19 * 4 + 2 = 78
position 7 -> 19 * 4 + 3 = 79
```

These values form the sequence's `slot_mapping`.

The block table is not a list of token IDs. It is a lookup table from logical sequence blocks to physical KV-cache blocks. This allows sequences to use non-contiguous memory while attention still accesses their tokens in logical order.

A shared prefix can also cause multiple sequences to reference the same physical block. For example:

```text
Sequence A block_table = [12, 19]
Sequence B block_table = [12, 27]
```

Here, both sequences reuse block `12`, which contains their shared first four prompt tokens. The KV manager increments the reference count for block `12` and releases it only after both sequences stop using it.

---

## 6. How the engine works together

The real runtime pipeline is a combination of these pieces:

### SequenceState
Responsible for the semantics of a single request:

- what tokens it contains,
- where it has processed so far,
- how to append new tokens,
- how to map its tokens to slots and blocks.

### PagedKVManager
Responsible for the global memory system:

- where KV blocks live,
- whether a prefix can be reused,
- which blocks are free versus evictable,
- how memory is released when a sequence ends.

### ContinuousBatchEngine
Responsible for orchestration:

- queueing new requests,
- batching active sequences,
- updating metadata,
- invoking model inference,
- advancing generation,
- retiring finished sequences.

Together, they form a minimal but meaningful vLLM-like execution stack.

---

## 7. Correctness model and invariants

A working paged-attention engine depends on a few invariants:

1. `processed_len` must stay within sequence bounds.
2. `slot_mapping` must match the physical block and offset assigned to each token.
3. `block_table` must remain row-major and padded to the same width for every active sequence.
4. `kv_mgr.allocate()` must happen before using the token range in attention.
5. sequence IDs and active state must be reset consistently between runs.
6. finished sequences must release their blocks so the pool does not leak memory.

These invariants are easy to violate in a minimal implementation, and they are also the main place where runtime bugs appear.

---

## 8. Strengths of this implementation

This project is good as an educational runtime because it captures several important ideas cleanly:

- clear separation between request state and cache manager,
- explicit support for prefix reuse,
- simple scheduling logic for continuous batching,
- easy-to-follow attention metadata assembly,
- buffer reuse for better GPU efficiency,
- minimal but meaningful architecture inspired by production systems.

It is especially useful for learning how a paged-attention engine is organized without needing to understand the full complexity of vLLM.

---

## 9. What can be improved further

Even though the architecture is solid, there are several high-impact improvements to consider.

### 9.1 Better scheduler semantics

The current engine schedule is simple and effective, but it can be improved by:

- separating prefill and decode queues,
- prioritizing shorter jobs first,
- using admission control for long prompts,
- limiting the number of mixed prefill/decode jobs in one step,
- batching only compatible sequences together.

This reduces head-of-line blocking and improves throughput.

### 9.2 More robust block reuse policy

The current prefix-hashing logic is useful, but a stronger system would include:

- finer-grained prefix tracking,
- reuse priorities based on recency and frequency,
- better LRU/clock eviction policies,
- block compaction or defragmentation when memory becomes fragmented.

### 9.3 More realistic attention metadata flow

For production-grade performance, attention metadata should be:

- generated in a more compact, kernel-friendly format,
- aligned with the exact model implementation,
- validated strictly against runtime constraints,
- kept device-local to avoid unnecessary CPU syncs.

### 9.4 Lower-overhead CUDA path

The current GPU staging logic is already a step forward, but more could be done by:

- reducing Python-side loops,
- using fused indexing or metadata assembly ops,
- batching host-to-device copies more carefully,
- avoiding redundant zeroing or unnecessary tensor creation,
- keeping memory lifetime consistent across stepping loops.

---

## 10. Recommended next steps

These are the most useful optimizations to pursue next.

### 10.1 Separate prefill and decode scheduling

The biggest practical throughput gain often comes from separating prefill-heavy requests from decode-heavy sequences.

Why?

- prefill is compute-heavy and benefits from larger chunks,
- decode is memory-heavy and benefits from steady, small, repeated batches,
- mixing them poorly can underutilize GPU compute and increase latency.

A two-queue scheduler is a significant upgrade from a single mixed active set.

### 10.2 Reduce CPU-GPU synchronization

The engine should avoid sending small pieces of metadata back and forth between CPU and GPU in the hot path.

Good targets:

- keep tensors on the model device whenever possible,
- avoid `tolist()` in the loop for every generated token when GPU results can be handled directly,
- precompute static metadata whenever possible,
- limit `.cpu()` conversions to only required final outputs.

### 10.3 Tune batch size and chunk size

Throughput depends strongly on the chosen values for:

- `max_batch_size`,
- `chunk_size`,
- cache block count,
- sequence lifetime.

The best values are workload-dependent, and they should be measured with real benchmarks rather than guessed.

### 10.4 Benchmark before deeper optimization

Before moving to compile or graph capture, benchmark the engine with:

- single-sequence latency,
- batch throughput,
- memory usage,
- variance in decode time,
- prompt-prefill cost under different chunk sizes.

This makes it easier to know whether later optimizations are actually helping.

---

## 11. Future optimization tasks

This section captures the next-level improvements that are useful once the runtime is already correct and stable.

### 11.1 TorchCompile

`torch.compile` can improve performance by fusing operations and optimizing the execution graph for the model path.

Good first checks:

- compile only the model forward pass, not the whole engine loop,
- benchmark eager vs compiled performance,
- ensure dynamic shapes and metadata remain stable,
- verify memory reuse still works under compiled execution.

Important caveat: compiled mode is often best attempted after correctness and metadata layout are stable. Compile is not a first step for debugging.

### 11.2 CUDA Graph capture

CUDA graphs reduce launch overhead by capturing repeated GPU work and replaying it with minimal host-side overhead.

This is especially useful for steady-state decode loops where the same pattern repeats over and over.

Before enabling graph capture, ensure:

- batch shapes are stable enough,
- sequence counts do not vary wildly each step,
- metadata layout is consistent across iterations,
- dynamic memory reuse does not cause capture invalidation.

Graph capture is a strong throughput optimization, but it should come after the scheduler and memory path are already tuned.

### 11.3 Kernel-level fusion and model optimization

If the model path becomes the bottleneck, focus on:

- fused layernorm or attention variants,
- lower precision kernels,
- quantized KV caches,
- better chunking strategies for long sequences,
- combining smaller operations into fewer kernel launches.

### 11.4 Profile-driven optimization

Use profiling tools to answer the real bottlenecks:

- is the model compute-bound or memory-bound?
- are KV allocations dominating runtime?
- are Python loops or metadata generation taking too much time?
- is host-to-device copy overhead significant?

This turns optimization from guessing into measurement.

---

## 12. Practical summary

SimpleVLLM is an excellent minimal example of the core structure behind vLLM-like serving systems:

- `SequenceState` tracks each request's semantics,
- `PagedKVManager` governs global block allocation and reuse,
- `ContinuousBatchEngine` orchestrates scheduling and inference.

The implementation is already a strong educational foundation, and the next major performance wins are likely to come from:

- better prefill/decode separation,
- tighter GPU staging and metadata flow,
- profiling-guided tuning,
- `torch.compile`,
- CUDA graph capture after the runtime is stable.

This is the right path from a readable prototype to a more production-like LLM serving runtime.

---

## 13. To-do list

- [ ] Benchmark eager throughput with different `max_batch_size` and `chunk_size`
- [ ] Separate prefill and decode scheduler queues
- [ ] Reduce CPU-GPU synchronization in the hot path
- [ ] Improve prefix reuse and eviction policy
- [ ] Add profiling instrumentation to measure bottlenecks
- [ ] Benchmark `torch.compile` on the model forward path
- [ ] Add CUDA graph capture for steady decode loops
- [ ] Validate memory stability under long-running batches
- [ ] Tune block sizing and cache capacity for realistic workloads
- [ ] Explore fused attention and lower-precision execution paths

---

## 14. Final note

The most important idea to remember is that this runtime is not only about token generation. It is about managing state, memory, and scheduling in a way that keeps GPUs busy while preserving correctness. That is the real lesson behind paged attention and continuous batching.

If you understand `SequenceState`, `PagedKVManager`, and `ContinuousBatchEngine`, you understand the core operating model of a modern serving stack, even at a small scale.
