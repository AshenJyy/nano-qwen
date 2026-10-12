"""
Benchmarking custom explicit attention implementation vs native PyTorch SDPA
on NVIDIA RTX 4060 Mobile (8GB VRAM).
"""

import math
import torch
import torch.nn.functional as F

from model import Qwen2Config, Qwen2ForCausalLM
from load_and_verify import load_model_from_gguf, GGUF_PATH, DEVICE, DTYPE


def explicit_attention_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
    """
    Explicit scaled dot-product attention calculation:
    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d) + mask) * V
    """
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    # Q * K^T
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    if mask is not None:
        scores = scores + mask

    # fp32 softmax for numerical stability across deep layers
    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

    # Context projection
    context = torch.matmul(attn_weights, v)
    return context, attn_weights


def profile_attention(model, prompt_tokens: int = 128, runs: int = 50):
    print("=" * 60)
    print(" Benchmarking Attention Implementation on RTX 4060 Laptop GPU")
    print("=" * 60)

    device = torch.device(DEVICE)
    dummy_input = torch.randint(100, 10000, (1, prompt_tokens), device=device)

    # GPU warmup
    for _ in range(5):
        _ = model(dummy_input)
    torch.cuda.synchronize()

    # 1. Native PyTorch Attention
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(runs):
        _ = model(dummy_input, custom_kernel_fn=None)
    end.record()
    torch.cuda.synchronize()

    native_latency = start.elapsed_time(end) / runs
    native_peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # 2. Custom Explicit Attention
    torch.cuda.reset_peak_memory_stats()
    start.record()
    for _ in range(runs):
        _ = model(dummy_input, custom_kernel_fn=explicit_attention_kernel)
    end.record()
    torch.cuda.synchronize()

    custom_latency = start.elapsed_time(end) / runs
    custom_peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"Test Configuration: Sequence Length = {prompt_tokens}, Sample Runs = {runs}\n")
    print(f"{'Implementation':<26} | {'Mean Latency':<14} | {'Peak VRAM':<14}")
    print("-" * 60)
    print(f"{'PyTorch Native SDPA':<26} | {native_latency:<11.2f} ms | {native_peak_vram:<11.1f} MB")
    print(f"{'Custom Explicit Attention':<26} | {custom_latency:<11.2f} ms | {custom_peak_vram:<11.1f} MB")
    print("-" * 60)
    print(f"Memory Overhead Delta: {abs(custom_peak_vram - native_peak_vram):.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    config = Qwen2Config()
    model = load_model_from_gguf(GGUF_PATH, config, device=DEVICE, dtype=DTYPE)
    profile_attention(model)
