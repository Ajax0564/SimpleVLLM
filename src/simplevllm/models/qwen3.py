import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from .config import QWEN3_CONFIG
from .utils import load_model_weights, load_weights_into_qwen

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale
        if self.shift is not None:
            norm_x = norm_x + self.shift
        return norm_x.to(input_dtype)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False, dtype=cfg["dtype"])
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False, dtype=cfg["dtype"])
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], bias=False, dtype=cfg["dtype"])

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))

def apply_rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

class GroupedQueryAttention(nn.Module):
    def __init__(self, layer_idx, cfg):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg["n_heads"]
        self.num_kv_groups = cfg["n_kv_groups"]
        self.head_dim = cfg["head_dim"]

        self.W_query = nn.Linear(cfg["emb_dim"], self.num_heads * self.head_dim, bias=False, dtype=cfg["dtype"])
        self.W_key = nn.Linear(cfg["emb_dim"],   self.num_kv_groups * self.head_dim, bias=False, dtype=cfg["dtype"])
        self.W_value = nn.Linear(cfg["emb_dim"],  self.num_kv_groups * self.head_dim, bias=False, dtype=cfg["dtype"])
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, cfg["emb_dim"], bias=False, dtype=cfg["dtype"])
        
        self.q_norm = RMSNorm(self.head_dim) if cfg.get("qk_norm") else None
        self.k_norm = RMSNorm(self.head_dim) if cfg.get("qk_norm") else None

    def forward(self, x, k_cache, v_cache, metadata):
        q = self.W_query(x).view(-1, self.num_heads, self.head_dim)
        k = self.W_key(x).view(-1, self.num_kv_groups, self.head_dim)
        v = self.W_value(x).view(-1, self.num_kv_groups, self.head_dim)
        
        if self.q_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        q = apply_rope(q, metadata['cos'], metadata['sin'])
        k = apply_rope(k, metadata['cos'], metadata['sin'])

        # Manual Paged Cache Update (Works well for compiled inference)
        slots = metadata['slot_mapping']
        b_idx = slots // metadata['block_size']
        o_idx = slots % metadata['block_size']
        
        k_cache[b_idx, o_idx] = k
        v_cache[b_idx, o_idx] = v

        if metadata['is_decoding']:
            # Decode Phase
            attn_out = flash_attn_with_kvcache(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=metadata['seqlens'],
                block_table=metadata['block_table'],
                causal=True
            )
        else:
            # Chunked Prefill / Mixed Batch Phase
            # Pass the global k_cache and v_cache along with the block_table
            attn_out = flash_attn_varlen_func(
                q, k_cache, v_cache,
                cu_seqlens_q=metadata['cu_seqlens_q'],
                cu_seqlens_k=metadata['cu_seqlens_k'],
                max_seqlen_q=metadata['max_seqlen_q'],
                max_seqlen_k=metadata['max_seqlen_k'],
                causal=True,
                block_table=metadata['block_table']
            )
        
        return self.out_proj(attn_out.view(-1, self.num_heads * self.head_dim))

class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.att = GroupedQueryAttention(layer_idx, cfg)
        self.ff = FeedForward(cfg)
        self.norm1, self.norm2 = RMSNorm(cfg["emb_dim"]), RMSNorm(cfg["emb_dim"])

    def forward(self, x, k_cache, v_cache, metadata):
        x = x + self.att(self.norm1(x), k_cache, v_cache, metadata)
        x = x + self.ff(self.norm2(x))
        return x

class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        inv_freq = 1.0 / (cfg["rope_base"] ** (torch.arange(0, cfg["head_dim"], 2).float() / cfg["head_dim"]))
        t = torch.arange(cfg["context_length"]).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_buf", freqs.cos().to(cfg["dtype"]))
        self.register_buffer("sin_buf", freqs.sin().to(cfg["dtype"]))

    def forward(self, input_ids, k_caches, v_caches, metadata):
        x = self.tok_emb(input_ids)
        for i, block in enumerate(self.trf_blocks):
            x = block(x, k_caches[i], v_caches[i], metadata)
        return self.out_head(self.final_norm(x))


def get_qwen3_model(weights_path):
    model = Qwen3Model(QWEN3_CONFIG)
    weights_dict = load_model_weights()
    load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
    return model