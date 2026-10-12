"""
nano-qwen: A clean, minimal from-scratch PyTorch implementation of Qwen2.5-1.5B.
Around 250 lines of self-contained Transformer code.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Qwen2Config:
    vocab_size: int = 151936
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2          # GQA group size = 12 // 2 = 6
    head_dim: int = 128                   # hidden_size // num_attention_heads
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0         # Qwen2.5 uses theta=1M for long context
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = True


class Qwen2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute variance in fp32 to prevent underflow
        orig_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return (normed * self.weight).to(orig_dtype)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 32768, theta: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.max_seq_len = max_seq_len

        # inv_freq: theta ^ (-2i / dim)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]  # [1, 1, seq_len, dim]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # (q * cos) + (rotate_half(q) * sin)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        # SwiGLU requires 3 projections: gate, up, down
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # down(silu(gate(x)) * up(x))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads       # 12
        self.head_dim = config.head_dim                   # 128
        self.num_kv_heads = config.num_key_value_heads   # 2
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # 6

        # Note: Qwen2 has bias=True on q, k, v projections, but bias=False on o_proj
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        custom_kernel_fn: Optional[callable] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, _ = x.shape

        # [bsz, seq_len, hidden] -> [bsz, heads, seq_len, head_dim]
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # repeat kv heads to match query heads (GQA: 2 heads -> 12 heads)
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        if custom_kernel_fn is not None:
            out, attn_weights = custom_kernel_fn(q, k, v, mask)
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale
            if mask is not None:
                scores = scores + mask
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(attn_weights, v)

        # [bsz, heads, seq_len, head_dim] -> [bsz, seq_len, hidden]
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(out), attn_weights


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2Attention(config)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen2MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        custom_kernel_fn: Optional[callable] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # pre-norm self-attention
        residual = x
        normed = self.input_layernorm(x)
        attn_out, weights = self.self_attn(normed, cos, sin, mask, custom_kernel_fn=custom_kernel_fn)
        x = residual + attn_out

        # pre-norm mlp
        residual = x
        normed = self.post_attention_layernorm(x)
        x = residual + self.mlp(normed)
        return x, weights


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)

    def forward(
        self,
        input_ids: torch.Tensor,
        custom_kernel_fn: Optional[callable] = None,
    ) -> Tuple[torch.Tensor, list]:
        bsz, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)

        cos, sin = self.rotary_emb(seq_len, device=x.device)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype)
            mask = torch.triu(mask, diagonal=1)[None, None, :, :]
        else:
            mask = None

        all_hidden_states = []
        for layer in self.layers:
            all_hidden_states.append(x)
            x, _ = layer(x, cos, sin, mask, custom_kernel_fn=custom_kernel_fn)

        x = self.norm(x)
        all_hidden_states.append(x)
        return x, all_hidden_states


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        custom_kernel_fn: Optional[callable] = None,
    ) -> Tuple[torch.Tensor, list]:
        hiddens, all_hiddens = self.model(input_ids, custom_kernel_fn=custom_kernel_fn)
        logits = self.lm_head(hiddens)
        return logits, all_hiddens

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_k: int = 50,
        custom_kernel_fn: Optional[callable] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self.forward(input_ids, custom_kernel_fn=custom_kernel_fn)
            next_token_logits = logits[:, -1, :]

            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            # eos check (151643 for Qwen2.5)
            if next_token.item() in (151643, 151645):
                break
        return input_ids


if __name__ == "__main__":
    cfg = Qwen2Config()
    model = Qwen2ForCausalLM(cfg)
    params = sum(p.numel() for p in model.parameters())
    print(f"nano-qwen instantiated: {params:,} parameters")
    print(f"config: layers={cfg.num_hidden_layers}, hidden_dim={cfg.hidden_size}, heads={cfg.num_attention_heads}/{cfg.num_key_value_heads}")
