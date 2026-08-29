"""
Export, packaging, and optimization script for the Draft Transformer model.
Saves model weights along with exact configuration metadata for seamless backend loading.
Supports automatic key-mapping for both nested and flat ModuleDict checkpoints.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch

from training.draft_arch import DraftTransformer, DraftTransformerConfig

KEY_MAP = {
    "norm1": "input_layernorm",
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "norm2": "post_attention_layernorm",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj"
}

def remap_state_dict(state_dict):
    """Translates state dict keys between flat ModuleDict and nested architectures."""
    new_sd = {}
    for k, v in state_dict.items():
        parts = k.split(".")
        if len(parts) >= 3 and parts[0] == "layers":
            layer_idx = parts[1]
            submod = parts[2]
            suffix = ".".join(parts[3:]) if len(parts) > 3 else "weight"
            if submod in KEY_MAP:
                target_sub = KEY_MAP[submod]
                new_key = f"layers.{layer_idx}.{target_sub}.{suffix}"
                new_sd[new_key] = v
                continue
        new_sd[k] = v
    return new_sd

def export_checkpoint(
    model: DraftTransformer,
    output_path: Path,
    tokenizer_name: str = "NousResearch/Meta-Llama-3.1-8B-Instruct"
):
    """Exports model weights, config, and metadata into a single deployment package."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "config": model.config.to_dict() if hasattr(model.config, "to_dict") else model.config,
        "state_dict": model.state_dict(),
        "tokenizer_name": tokenizer_name,
        "vocab_size": model.config.vocab_size if hasattr(model.config, "vocab_size") else 128256,
        "params_count": model.count_parameters()
    }
    
    torch.save(payload, str(output_path))
    print(f"[OK] Model successfully packaged and exported to: {output_path}")

def load_exported_checkpoint(checkpoint_path: Path, device: str = "cpu") -> DraftTransformer:
    """Loads an exported checkpoint into a ready-to-infer DraftTransformer instance."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    payload = torch.load(str(checkpoint_path), map_location=device)
    raw_config = payload.get("config", {})
    if isinstance(raw_config, dict):
        config = DraftTransformerConfig.from_dict(raw_config)
    else:
        config = raw_config

    model = DraftTransformer(config)
    raw_sd = payload["state_dict"]
    mapped_sd = remap_state_dict(raw_sd)
    model.load_state_dict(mapped_sd, strict=False)
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and Package Draft Transformer")
    parser.add_argument("--output", type=str, default="checkpoints/draft_model.pt", help="Path to save exported model")
    args = parser.parse_args()

    config = DraftTransformerConfig()
    model = DraftTransformer(config)
    export_checkpoint(model, Path(args.output))
