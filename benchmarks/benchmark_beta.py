import time
import gc
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..src.simplevllm.engine import ContinuousBatchEngine,PagedKVManager
from ..src.simplevllm.models import get_qwen3_model,Qwen3Config
torch.manual_seed(123)
device = "cuda" if torch.cuda.is_available() else "cpu"

def run_unified_benchmark(batch_sizes=[1, 2, 4, 8], max_new_tokens=1024, prompt_text="Generate a very long and detailed story about artificial intelligence."):
    custom_tps_results, hf_tps_results = [], []
    custom_mem_results, hf_mem_results = [], []

    print("Loading Hugging Face Qwen3-0.6B model...")
    hf_model_id = "Qwen/Qwen3-0.6B"
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2"
    )
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    # Prepare prompts
    custom_prompt_ids = hf_tokenizer.encode(prompt_text)
    hf_input_ids = hf_tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda")
    
    # Initialize a new KV Manager with enough blocks for BS=8 (128 blocks)
    local_mgr = PagedKVManager(Qwen3Config, max_blocks=128, device="cuda")
    model = get_qwen3_model()
    model.to('cuda')
    bench_engine = ContinuousBatchEngine(model, local_mgr, Qwen3Config)

    print("--- WARMUP PHASE ---")
    # Custom Engine warmup
    bench_engine.reset()
    bench_engine.add_sequence(custom_prompt_ids, max_gen_len=150)
    while bench_engine.waiting_room or bench_engine.active:
        bench_engine.step()

    # HF Engine warmup
    with torch.inference_mode():
        _ = hf_model.generate(hf_input_ids, max_new_tokens=150, do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("Warmup complete.\n")

    print(f"Starting Unified Benchmark (Gen Length: {max_new_tokens})...")
    for bs in batch_sizes:
        print(f"\n--- Testing Batch Size: {bs} ---")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        bench_engine.reset()

        for _ in range(bs):
            bench_engine.add_sequence(custom_prompt_ids, max_gen_len=max_new_tokens)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.time()
        total_custom_tokens = 0

        while bench_engine.waiting_room or bench_engine.active:
            finished = bench_engine.step()
            for sid, tokens in finished.items():
                total_custom_tokens += len(tokens) - len(custom_prompt_ids)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()
        custom_peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        custom_duration = end_time - start_time
        custom_tps = total_custom_tokens / custom_duration

        custom_tps_results.append(custom_tps)
        custom_mem_results.append(custom_peak_mem)
        print(f"[Custom Engine] TPS: {custom_tps:6.2f} tokens/s | Peak VRAM: {custom_peak_mem:.2f} GB")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        batch_input_ids = hf_input_ids.repeat(bs, 1)
        attention_mask = torch.ones_like(batch_input_ids)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.time()

        with torch.inference_mode():
            _ = hf_model.generate(
                batch_input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()
        hf_peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        hf_duration = end_time - start_time
        total_hf_tokens = bs * max_new_tokens
        hf_tps = total_hf_tokens / hf_duration

        hf_tps_results.append(hf_tps)
        hf_mem_results.append(hf_peak_mem)
        print(f"[HF model.gen]  TPS: {hf_tps:6.2f} tokens/s | Peak VRAM: {hf_peak_mem:.2f} GB")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Subplot 1: Throughput ---
    ax1.plot(batch_sizes, custom_tps_results, marker='o', linestyle='-', color='b', linewidth=2, label='Custom Engine')
    ax1.plot(batch_sizes, hf_tps_results, marker='s', linestyle='--', color='r', linewidth=2, label='HF model.generate')
    ax1.set_title(f'Throughput Comparison (Tokens/Sec)')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Throughput')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(batch_sizes)
    ax1.legend()

    max_tps = max(max(custom_tps_results), max(hf_tps_results))
    for x, y_c, y_h in zip(batch_sizes, custom_tps_results, hf_tps_results):
        if y_c >= y_h:
            ax1.text(x, y_c + (max_tps * 0.02), f"{y_c:.1f}", ha='center', va='bottom', color='b', fontweight='bold')
            ax1.text(x, y_h - (max_tps * 0.03), f"{y_h:.1f}", ha='center', va='top', color='r', fontweight='bold')
        else:
            ax1.text(x, y_c - (max_tps * 0.03), f"{y_c:.1f}", ha='center', va='top', color='b', fontweight='bold')
            ax1.text(x, y_h + (max_tps * 0.02), f"{y_h:.1f}", ha='center', va='bottom', color='r', fontweight='bold')

    ax2.plot(batch_sizes, custom_mem_results, marker='o', linestyle='-', color='b', linewidth=2, label='Custom Engine')
    ax2.plot(batch_sizes, hf_mem_results, marker='s', linestyle='--', color='r', linewidth=2, label='HF model.generate')
    ax2.set_title(f'Peak VRAM Usage Comparison (GB)')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Peak VRAM (GB)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xticks(batch_sizes)
    ax2.legend()

    max_mem = max(max(custom_mem_results), max(hf_mem_results))
    for x, y_c, y_h in zip(batch_sizes, custom_mem_results, hf_mem_results):
        if y_c >= y_h:
            ax2.text(x, y_c + (max_mem * 0.005), f"{y_c:.2f}", ha='center', va='bottom', color='b', fontweight='bold')
            ax2.text(x, y_h - (max_mem * 0.005), f"{y_h:.2f}", ha='center', va='top', color='r', fontweight='bold')
        else:
            ax2.text(x, y_c - (max_mem * 0.005), f"{y_c:.2f}", ha='center', va='top', color='b', fontweight='bold')
            ax2.text(x, y_h + (max_mem * 0.005), f"{y_h:.2f}", ha='center', va='bottom', color='r', fontweight='bold')

    plt.suptitle(f"Benchmark Results (Generation Length: {max_new_tokens})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Clean up HF model to free memory if needed afterwards
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

# Run the unified benchmark (using a length of 1024 for reasonable speed, change if needed)
run_unified_benchmark(batch_sizes=[1, 2, 4, 8], max_new_tokens=1024)
