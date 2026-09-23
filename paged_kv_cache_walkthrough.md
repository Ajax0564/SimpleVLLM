# Paged KV Cache Execution Walkthrough & Technical Reference

### Execution Walkthrough Setup

To demonstrate how memory blocks, slot mappings, block tables, and paged KV caches operate in the inference engine, we establish a minimal environment configuration and two concrete sample inputs.

#### Configuration Parameters
* **`block_size`**: $4$ tokens per physical memory block.
* **`max_blocks`**: $8$ total physical blocks allocated in GPU memory.
* **`n_kv_groups`**: $2$ KV heads.
* **`head_dim`**: $16$ dimensions per head.
* **`n_layers`**: $1$ layer (for clarity).
* **Initial `free_blocks` Queue**: `deque([0, 1, 2, 3, 4, 5, 6, 7])`.

#### Sample Inputs
* **Sequence 0 (`s0`)**: Prompt length $N_0 = 7$ tokens $\rightarrow$ `[101, 102, 103, 104, 105, 106, 107]`.
* **Sequence 1 (`s1`)**: Prompt length $N_1 = 3$ tokens $\rightarrow$ `[201, 202, 203]`.

---

### Core Data Structures & Mathematical Mapping

1. **Physical Cache Tensor (`PagedKVManager.k_cache` / `v_cache`)**
   Shape: $(\text{max\_blocks}, \text{block\_size}, \text{n\_kv\_groups}, \text{head\_dim}) = (8, 4, 2, 16)$.
   Each block holds $4$ token slots. The physical slot index in memory is calculated as:
   $$\text{slot\_mapping}[t] = \text{physical\_block\_id} \times \text{block\_size} + \text{offset}$$

2. **Block Table (`SequenceState.block_table`)**
   A 1D tensor per sequence mapping a logical sequence block index to a physical memory block index.
   $$\text{b\_idx} = \lfloor t / \text{block\_size} \rfloor, \quad \text{o\_idx} = t \pmod{\text{block\_size}}$$
   $$\text{physical\_block\_id} = \text{block\_table}[\text{b\_idx}]$$

3. **Slot Mapping (`SequenceState.slot_mapping`)**
   A 1D tensor mapping each token position $t$ in the sequence to its flattened location inside the global KV cache memory pool.

---

### Execution Step 1: Scheduling & Memory Allocation

When `engine.add_sequence()` and `engine.step()` run, the engine schedules sequences from `waiting_room` to `active`.

#### Allocation Calculation (`PagedKVManager.allocate`)
* **`s0`** requires $\lceil 7 / 4 \rceil = 2$ blocks.
  * Pops `0` and `1` from `free_blocks`.
  * `s0.block_table` $\rightarrow$ `[0, 1, 0, 0, 0, 0, 0, 0]`.
* **`s1`** requires $\lceil 3 / 4 \rceil = 1$ block.
  * Pops `2` from `free_blocks`.
  * `s1.block_table` $\rightarrow$ `[2, 0, 0, 0, 0, 0, 0, 0]`.

**Remaining `free_blocks`**: `deque([3, 4, 5, 6, 7])`.

---

### Execution Step 2: The Prefill Phase (Prompt Processing)

Both sequences enter the pipeline with `is_prefill = True`.

#### 1. Token-to-Slot Mapping Generation (`SequenceState.update_metadata`)

##### Sequence 0 (`s0`): Tokens $0 \dots 6$
| Token Index ($t$) | Token ID | Logical Block (`b_idx`) | Block Offset (`o_idx`) | Assigned Physical Block | Computed Slot (`slot_mapping[t]`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | 101 | $\lfloor 0/4 \rfloor = 0$ | $0 \pmod 4 = 0$ | `block_table[0] = 0` | $0 \times 4 + 0 = \mathbf{0}$ |
| **1** | 102 | $\lfloor 1/4 \rfloor = 0$ | $1 \pmod 4 = 1$ | `block_table[0] = 0` | $0 \times 4 + 1 = \mathbf{1}$ |
| **2** | 103 | $\lfloor 2/4 \rfloor = 0$ | $2 \pmod 4 = 2$ | `block_table[0] = 0` | $0 \times 4 + 2 = \mathbf{2}$ |
| **3** | 104 | $\lfloor 3/4 \rfloor = 0$ | $3 \pmod 4 = 3$ | `block_table[0] = 0` | $0 \times 4 + 3 = \mathbf{3}$ |
| **4** | 105 | $\lfloor 4/4 \rfloor = 1$ | $4 \pmod 4 = 0$ | `block_table[1] = 1` | $1 \times 4 + 0 = \mathbf{4}$ |
| **5** | 106 | $\lfloor 5/4 \rfloor = 1$ | $5 \pmod 4 = 1$ | `block_table[1] = 1` | $1 \times 4 + 1 = \mathbf{5}$ |
| **6** | 107 | $\lfloor 6/4 \rfloor = 1$ | $6 \pmod 4 = 2$ | `block_table[1] = 1` | $1 \times 4 + 2 = \mathbf{6}$ |

##### Sequence 1 (`s1`): Tokens $0 \dots 2$
| Token Index ($t$) | Token ID | Logical Block (`b_idx`) | Block Offset (`o_idx`) | Assigned Physical Block | Computed Slot (`slot_mapping[t]`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | 201 | $\lfloor 0/4 \rfloor = 0$ | $0 \pmod 4 = 0$ | `block_table[0] = 2` | $2 \times 4 + 0 = \mathbf{8}$ |
| **1** | 202 | $\lfloor 1/4 \rfloor = 0$ | $1 \pmod 4 = 1$ | `block_table[0] = 2` | $2 \times 4 + 1 = \mathbf{9}$ |
| **2** | 203 | $\lfloor 2/4 \rfloor = 0$ | $2 \pmod 4 = 2$ | `block_table[0] = 2` | $2 \times 4 + 2 = \mathbf{10}$ |

#### 2. Inference Metadata Preparation (`_prepare_inference_data`)
The engine flattens token data for variable-length batching:
* **`input_ids`**: `[101, 102, 103, 104, 105, 106, 107, 201, 202, 203]` (1D Tensor of length 10).
* **`slot_mapping`**: `[0, 1, 2, 3, 4, 5, 6, 8, 9, 10]`.
* **`cu_seqlens`**: `[0, 7, 10]` (Cumulative sequence lengths).
* **`is_decoding`**: `False`.

#### 3. Attention Execution & Cache Writing (`GroupedQueryAttention.forward`)
1. $K$ and $V$ projections calculate computed keys and values for all 10 tokens.
2. The model resolves target slots in the physical cache tensor:
   ```python
   b_idx = slots // metadata['block_size']  # [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
   o_idx = slots % metadata['block_size']   # [0, 1, 2, 3, 0, 1, 2, 0, 1, 2]
   k_cache[b_idx, o_idx] = k
   v_cache[b_idx, o_idx] = v
   ```
3. `flash_attn_varlen_func` executes causal attention over prompt tokens.
4. Logits sample new tokens at the sequence boundaries:
   * **`s0`** samples token ID **`108`** at index $t=7$.
   * **`s1`** samples token ID **`204`** at index $t=3$.
5. Sequence states transition to decoding mode (`s0.is_prefill = False`, `s1.is_prefill = False`).

---

### Execution Step 3: Decode Phase (First Generated Token)

In the second invocation of `engine.step()`, both sequences process $1$ single query token each.

* **Input Tokens**: `[108]` for `s0` (at index $t=7$), `[204]` for `s1` (at index $t=3$).
* **Current Total Token Lengths**: `s0.num_tokens = 8`, `s1.num_tokens = 4`.

#### 1. Metadata Generation (`update_metadata` for Decoding)
When decoding, `start = s.num_tokens - 1`:

* **`s0` Token 7**:
  $$\text{b\_idx} = \lfloor 7/4 \rfloor = 1, \quad \text{o\_idx} = 7 \pmod 4 = 3$$
  $$\text{Physical Block} = \text{s0.block\_table}[1] = 1$$
  $$\text{slot\_mapping}[7] = 1 \times 4 + 3 = \mathbf{7}$$

* **`s1` Token 3**:
  $$\text{b\_idx} = \lfloor 3/4 \rfloor = 0, \quad \text{o\_idx} = 3 \pmod 4 = 3$$
  $$\text{Physical Block} = \text{s1.block\_table}[0] = 2$$
  $$\text{slot\_mapping}[3] = 2 \times 4 + 3 = \mathbf{11}$$

#### 2. Inference Metadata (`_prepare_inference_data`)
* **`input_ids`**: `[108, 204]` (Shape: $2$).
* **`slot_mapping`**: `[7, 11]`.
* **`seqlens`**: `[8, 4]` (Total context length of each sequence).
* **`block_table`**: Stacked 2D tensor of shape $(2, \text{max\_blocks})$:
  $$\begin{bmatrix} 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \end{bmatrix}$$
* **`is_decoding`**: `True`.

#### 3. Paged Attention Execution
1. `GroupedQueryAttention` writes the single generated Key/Value vector for each sequence directly into physical slots:
   * `s0` key/value written to `k_cache[1, 3]` and `v_cache[1, 3]`.
   * `s1` key/value written to `k_cache[2, 3]` and `v_cache[2, 3]`.
2. `flash_attn_with_kvcache` is called:
   ```python
   flash_attn_with_kvcache(
       q.unsqueeze(1), k_cache, v_cache,
       cache_seqlens=metadata['seqlens'],     # [8, 4]
       block_table=metadata['block_table'],   # [[0, 1, ...], [2, 0, ...]]
       causal=True
   )
   ```
   FlashAttention uses `block_table` to perform dynamic memory lookup, gathering $K/V$ vectors from physical blocks `[0, 1]` for sequence 0 (length 8) and block `[2]` for sequence 1 (length 4).
3. Logits sample next tokens:
   * **`s0`** samples token ID **`109`** (becomes token index $t=8$).
   * **`s1`** samples token ID **`205`** (becomes token index $t=4$).

---

### Execution Step 4: Dynamic Block Allocation Boundary Crossing

In the third invocation of `engine.step()`, both sequences exceed their currently allocated block capacities.

#### Dynamic Block Allocation Trigger (`kv_mgr.allocate`)
1. **`s0`**: `num_tokens = 9` (Index $t=8$).
   $$\text{Needed Blocks} = \lceil 9 / 4 \rceil = 3 \text{ blocks}$$
   Currently has $2$ blocks (`block_count = 2`).
   * Engine pops block **`3`** from `free_blocks`.
   * `s0.block_table` updated $\rightarrow$ `[0, 1, 3, 0, 0, 0, 0, 0]`.
2. **`s1`**: `num_tokens = 5` (Index $t=4$).
   $$\text{Needed Blocks} = \lceil 5 / 4 \rceil = 2 \text{ blocks}$$
   Currently has $1$ block (`block_count = 1`).
   * Engine pops block **`4`** from `free_blocks`.
   * `s1.block_table` updated $\rightarrow$ `[2, 4, 0, 0, 0, 0, 0, 0]`.

**Remaining `free_blocks`**: `deque([5, 6, 7])`.

#### Slot Calculation for Boundary Tokens
* **`s0` Token 8**:
  $$\text{b\_idx} = \lfloor 8/4 \rfloor = 2, \quad \text{o\_idx} = 8 \pmod 4 = 0$$
  $$\text{Physical Block} = \text{s0.block\_table}[2] = 3$$
  $$\text{slot\_mapping}[8] = 3 \times 4 + 0 = \mathbf{12}$$

* **`s1` Token 4**:
  $$\text{b\_idx} = \lfloor 4/4 \rfloor = 1, \quad \text{o\_idx} = 4 \pmod 4 = 0$$
  $$\text{Physical Block} = \text{s1.block\_table}[1] = 4$$
  $$\text{slot\_mapping}[4] = 4 \times 4 + 0 = \mathbf{16}$$

---

### Complete Physical KV Cache Memory Layout Matrix

The table below illustrates the precise physical memory layout of the global KV Cache tensor across physical blocks $0 \dots 4$ after Step 4.

| Physical Block ID | Offset 0 (`o_idx=0`) | Offset 1 (`o_idx=1`) | Offset 2 (`o_idx=2`) | Offset 3 (`o_idx=3`) | Sequence Assignment |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Block 0** (Slots 0–3) | `s0` Token 0 (ID 101) | `s0` Token 1 (ID 102) | `s0` Token 2 (ID 103) | `s0` Token 3 (ID 104) | Sequence 0 (Block 0) |
| **Block 1** (Slots 4–7) | `s0` Token 4 (ID 105) | `s0` Token 5 (ID 106) | `s0` Token 6 (ID 107) | `s0` Token 7 (ID 108) | Sequence 0 (Block 1) |
| **Block 2** (Slots 8–11) | `s1` Token 0 (ID 201) | `s1` Token 1 (ID 202) | `s1` Token 2 (ID 203) | `s1` Token 3 (ID 204) | Sequence 1 (Block 0) |
| **Block 3** (Slots 12–15) | `s0` Token 8 (ID 109) | *Unused Slot* | *Unused Slot* | *Unused Slot* | Sequence 0 (Block 2) |
| **Block 4** (Slots 16–19) | `s1` Token 4 (ID 205) | *Unused Slot* | *Unused Slot* | *Unused Slot* | Sequence 1 (Block 1) |
| **Block 5–7** | *Free Memory* | *Free Memory* | *Free Memory* | *Free Memory* | In `free_blocks` queue |