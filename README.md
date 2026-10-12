# nano-qwen

A standalone PyTorch implementation of Qwen2.5-1.5B in ~250 lines of code.

Loads weights directly from local GGUF files without Hugging Face `transformers` modeling code. Runs comfortably on consumer hardware (tested on RTX 4060 Laptop GPU, ~3.5 GB peak VRAM in BF16).

## Overview

- **`model.py`**: Complete Qwen2.5 architecture (~250 LOC). Implements Grouped Query Attention (12 Q heads, 2 KV heads), RoPE ($\theta=10^6$), SwiGLU MLP, and RMSNorm.
- **`load_and_verify.py`**: Parses GGUF tensors, maps them into `model.py`, and runs text generation.
- **`custom_attention.py`**: Explicit attention implementation benchmarked against PyTorch SDPA.
- **`latent_probe_pca.py`**: Extracts hidden activations across layers 0, 14, and 27, and projects them with PCA.

## Quickstart

```bash
git clone https://github.com/AshenJyy/nano-qwen.git
cd nano-qwen

# setup environment
uv venv .venv --python 3.12
.venv\Scripts\activate   # on Linux/macOS: source .venv/bin/activate

# dependencies
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install numpy gguf transformers scikit-learn matplotlib
```

Run generation:
```bash
python load_and_verify.py
```

Benchmark attention:
```bash
python custom_attention.py
```

Plot layer activations:
```bash
python latent_probe_pca.py
```

## Model Architecture

Matches the official Qwen2.5-1.5B specification:

| Parameter | Value |
|---|---|
| Vocabulary Size | 151,936 |
| Hidden Dimension | 1,536 |
| Layers | 28 |
| Attention Heads | 12 Q heads, 2 KV heads (GQA, 6:1) |
| Head Dimension | 128 |
| Intermediate Size | 8,960 (SwiGLU) |
| RoPE Base ($\theta$) | 1,000,000 |
| Weight Tying | True |
| Total Parameters | ~1.54B |

### Memory

In BF16, model weights take ~2.88 GB. Total peak VRAM during inference sits around ~3.44 GB, leaving ample headroom on an 8 GB GPU.

### Attention Benchmark (seq_len=128, RTX 4060 Mobile)

| Implementation | Latency | Memory Overhead Delta |
|---|---|---|
| PyTorch Native SDPA | 65.80 ms | Baseline |
| Custom Explicit Attention | 65.82 ms | 0.00 MB |

### Hidden Representation Across Layers

Hidden states extracted at layers 0, 14, and 27 across 15 prompts in 3 categories (Systems/Compilers, Philosophy/Ethics, Daily Life), projected to 2D via PCA:

![layer representations](figures/latent_evolution_3panel.png)

Early layers (layer 0) reflect mostly lexical/token-level features with overlapping distributions. By middle and final layers (layers 14 and 27), representations separate into distinct semantic clusters.

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) by Alibaba Cloud

## License

MIT
