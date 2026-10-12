"""
Probes latent space activations across early, middle, and late layers of Qwen2.5-1.5B,
and projects the 1536-dimensional representations to 2D via PCA.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoTokenizer

from model import Qwen2Config, Qwen2ForCausalLM
from load_and_verify import load_model_from_gguf, GGUF_PATH, DEVICE, DTYPE

# 15 test prompts across 3 distinct semantic domains
PROMPTS = [
    # Systems & Compilers
    ("GPU kernel thread blocks coordinate via shared memory SRAM.", "Systems/Compilers"),
    ("Compilers optimize abstract syntax trees through register allocation.", "Systems/Compilers"),
    ("High-bandwidth memory latency is masked by parallel warp scheduling.", "Systems/Compilers"),
    ("Cache coherence protocols synchronize multi-core hardware states.", "Systems/Compilers"),
    ("PTX assembly instructions map directly onto streaming multiprocessors.", "Systems/Compilers"),

    # Philosophy & Ethics
    ("Does artificial general intelligence imply moral agency and consciousness?", "Philosophy/Ethics"),
    ("Existential dread emerges when autonomous systems displace human talent.", "Philosophy/Ethics"),
    ("The ethics of algorithmic governance in post-scarcity civilizations.", "Philosophy/Ethics"),
    ("Free will dissolves under deterministic physical laws of nature.", "Philosophy/Ethics"),
    ("Human dignity is challenged by automated cognitive machines.", "Philosophy/Ethics"),

    # Everyday Life & Culinary
    ("Freshly baked sourdough bread requires slow overnight fermentation.", "Daily Life/Culinary"),
    ("Crispy chocolate chip cookies are best served with cold milk.", "Daily Life/Culinary"),
    ("A weekend hike through pine forests brings refreshing quietness.", "Daily Life/Culinary"),
    ("Brewing aromatic espresso requires exact water pressure and temperature.", "Daily Life/Culinary"),
    ("Packing a light tent and sleeping bag for summer mountain camping.", "Daily Life/Culinary"),
]

DOMAIN_COLORS = {
    "Systems/Compilers": "#1f77b4",       # Blue
    "Philosophy/Ethics": "#d62728",       # Red
    "Daily Life/Culinary": "#2ca02c",     # Green
}


def extract_layer_representations(model, tokenizer, prompts):
    target_layers = [0, 14, 27]  # Input layer, mid layer, final layer
    activations = {l: [] for l in target_layers}
    labels = []
    domains = []

    print(f"Extracting activation vectors for {len(prompts)} prompts...")
    for text, domain in prompts:
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            _, all_hiddens = model(input_ids)

        for l in target_layers:
            layer_tensor = all_hiddens[l + 1][0].to(torch.float32)  # [seq_len, 1536]
            pooled = layer_tensor.mean(dim=0).cpu().numpy()
            activations[l].append(pooled)

        labels.append(text)
        domains.append(domain)

    return activations, domains, target_layers


def plot_pca_evolution(activations, domains, target_layers, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    print("Computing PCA projections and saving figures...")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    layer_titles = {
        0: "Layer 0 (Shallow / Lexical)",
        14: "Layer 14 (Intermediate / Syntactic)",
        27: "Layer 27 (Deep / Abstract Semantic)"
    }

    for idx, l in enumerate(target_layers):
        X = np.array(activations[l])
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        ax = axes[idx]
        for domain, color in DOMAIN_COLORS.items():
            mask = [d == domain for d in domains]
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=color,
                label=domain if idx == 0 else "",
                s=85,
                alpha=0.85,
                edgecolors="black",
                linewidth=0.7
            )

        var = pca.explained_variance_ratio_
        ax.set_title(
            f"{layer_titles[l]}\nExplained Variance: (PC1: {var[0]*100:.1f}%, PC2: {var[1]*100:.1f}%)",
            fontsize=11,
            fontweight="bold"
        )
        ax.set_xlabel("Principal Component 1 (PC1)", fontsize=10)
        ax.set_ylabel("Principal Component 2 (PC2)", fontsize=10)

    fig.suptitle(
        "Latent Vector Space Manifold Evolution Across Qwen2.5-1.5B (RTX 4060 Mobile)",
        fontsize=13,
        fontweight="bold",
        y=1.02
    )
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True, fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "latent_evolution_3panel.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to: {out_path}")


def main():
    config = Qwen2Config()
    model = load_model_from_gguf(GGUF_PATH, config, device=DEVICE, dtype=DTYPE)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)

    activations, domains, target_layers = extract_layer_representations(model, tokenizer, PROMPTS)
    plot_pca_evolution(activations, domains, target_layers, output_dir=r"E:\my-qwen-bedrock\figures")


if __name__ == "__main__":
    main()
