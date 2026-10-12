"""
Loads weights directly from a local GGUF file into nano-qwen,
and runs a quick greedy text generation check.
"""

import time
import torch
from gguf import GGUFReader
from transformers import AutoTokenizer

from model import Qwen2Config, Qwen2ForCausalLM

GGUF_PATH = r"E:\Qwen2.5-1.5B-Instruct\models\qwen2.5-1.5b-instruct-fp16.gguf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def build_gguf_name_map(num_layers: int = 28):
    """Maps GGUF tensor names to our custom PyTorch state_dict."""
    mapping = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    for i in range(num_layers):
        mapping[f"blk.{i}.attn_norm.weight"] = f"model.layers.{i}.input_layernorm.weight"
        mapping[f"blk.{i}.attn_q.weight"] = f"model.layers.{i}.self_attn.q_proj.weight"
        mapping[f"blk.{i}.attn_q.bias"] = f"model.layers.{i}.self_attn.q_proj.bias"
        mapping[f"blk.{i}.attn_k.weight"] = f"model.layers.{i}.self_attn.k_proj.weight"
        mapping[f"blk.{i}.attn_k.bias"] = f"model.layers.{i}.self_attn.k_proj.bias"
        mapping[f"blk.{i}.attn_v.weight"] = f"model.layers.{i}.self_attn.v_proj.weight"
        mapping[f"blk.{i}.attn_v.bias"] = f"model.layers.{i}.self_attn.v_proj.bias"
        mapping[f"blk.{i}.attn_output.weight"] = f"model.layers.{i}.self_attn.o_proj.weight"
        mapping[f"blk.{i}.ffn_norm.weight"] = f"model.layers.{i}.post_attention_layernorm.weight"
        mapping[f"blk.{i}.ffn_gate.weight"] = f"model.layers.{i}.mlp.gate_proj.weight"
        mapping[f"blk.{i}.ffn_up.weight"] = f"model.layers.{i}.mlp.up_proj.weight"
        mapping[f"blk.{i}.ffn_down.weight"] = f"model.layers.{i}.mlp.down_proj.weight"
    return mapping


def load_model_from_gguf(gguf_path: str, config: Qwen2Config, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    print(f"Initializing nano-qwen architecture...")
    model = Qwen2ForCausalLM(config)

    print(f"Reading tensors from GGUF binary: {gguf_path}")
    reader = GGUFReader(gguf_path)
    name_map = build_gguf_name_map(config.num_hidden_layers)

    state_dict = {}
    matched = 0

    for tensor in reader.tensors:
        if tensor.name in name_map:
            target_key = name_map[tensor.name]
            state_dict[target_key] = torch.from_numpy(tensor.data.copy()).to(dtype)
            matched += 1

    print(f"Mapped {matched} tensors directly into model state_dict.")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Note: {len(missing)} missing keys (expected if tied embeddings).")

    model = model.to(device=device, dtype=dtype)
    model.eval()

    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"Device: {torch.cuda.get_device_name(0)} | VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

    return model


def main():
    config = Qwen2Config()
    model = load_model_from_gguf(GGUF_PATH, config, device=DEVICE, dtype=DTYPE)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)

    prompt = "Explain what a GPU kernel is in one sentence."
    messages = [
        {"role": "user", "content": prompt}
    ]
    prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt_formatted, return_tensors="pt").input_ids.to(DEVICE)

    print(f"\nPrompt: {prompt}")
    print("Generating...")

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=48, temperature=0.7)
    elapsed = time.time() - t0

    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nResponse:\n{generated_text}\n")
    print(f"Elapsed: {elapsed:.2f}s | Tokens generated: {output_ids.shape[1] - input_ids.shape[1]}")


if __name__ == "__main__":
    main()
