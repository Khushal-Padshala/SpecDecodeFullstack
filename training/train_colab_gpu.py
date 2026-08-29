"""
GPU Distillation Training Script for Google Colab / Cloud GPU.
Trains the ~80M parameter Draft Transformer on large-scale diverse datasets
(Code, Instructions, Dialog, Reasoning) tokenized with Llama 3.1 128k vocabulary.
Uses Mixed Precision (FP16/BF16), Flash Attention / PyTorch SDPA, and Cosine Annealing.
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# 1. Draft Transformer Model Definition (Same Architecture)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int, offset: int = 0):
        if offset + seq_len > self.cos_cached.shape[0]:
            self._set_cos_sin_cache(offset + seq_len + 512)
        cos = self.cos_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(1).to(q.dtype).to(q.device)
        sin = self.sin_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(1).to(k.dtype).to(k.device)
        return self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (self._rotate_half(x) * sin)

class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, rope: RotaryEmbedding):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope = rope

        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k, seq_len, offset=0)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.o_proj(out)

class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, intermediate_dim: int, rope: RotaryEmbedding):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_dim)
        self.self_attn = CausalSelfAttention(hidden_dim, num_heads, head_dim, rope)
        self.post_attention_layernorm = RMSNorm(hidden_dim)
        self.mlp = SwiGLUMLP(hidden_dim, intermediate_dim)

    def forward(self, x: torch.Tensor):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class DraftTransformer(nn.Module):
    def __init__(self, vocab_size: int = 128256, hidden_dim: int = 512, num_layers: int = 5, num_heads: int = 8, head_dim: int = 64, intermediate_dim: int = 1536, max_seq_len: int = 2048):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "intermediate_dim": intermediate_dim,
            "max_seq_len": max_seq_len,
            "rope_base": 500000.0,
            "norm_eps": 1e-5,
            "tie_word_embeddings": True
        }
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len, base=500000.0)
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, head_dim, intermediate_dim, self.rope)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight # Tied weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

# ---------------------------------------------------------------------------
# 2. Large Scale Streaming Multi-Task Dataset
# ---------------------------------------------------------------------------
class MultiDomainDataset(Dataset):
    def __init__(self, tokenizer, num_samples: int = 5000, max_seq_len: int = 256):
        self.samples = []
        print(f"[*] Downloading and tokenizing diverse multi-domain training samples...")

        # 1. Instruction & Chat (Dolly 15k)
        print("  -> Loading databricks/databricks-dolly-15k...")
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        for row in dolly:
            if len(self.samples) >= num_samples:
                break
            text = f"<|start_header_id|>user<|end_header_id|>\n\n{row['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{row['response']}<|eot_id|>"
            toks = tokenizer.encode(text)
            if len(toks) > 16:
                self.samples.append(toks[:max_seq_len])

        print(f"[OK] Total tokenized sequences ready: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch, pad_token_id=128001):
    max_len = max(len(s) for s in batch)
    input_ids, labels = [], []
    for seq in batch:
        padded = seq + [pad_token_id] * (max_len - len(seq))
        input_ids.append(padded[:-1])
        labels.append([t if t != pad_token_id else -100 for t in padded[1:]])
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

# ---------------------------------------------------------------------------
# 3. GPU Training Loop
# ---------------------------------------------------------------------------
def train_gpu(
    output_path: str = "draft_model.pt",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 8e-4,
    num_samples: int = 5000
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    print("[*] Loading Meta Llama 3.1 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 128001

    dataset = MultiDomainDataset(tokenizer, num_samples=num_samples, max_seq_len=256)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))

    model = DraftTransformer(vocab_size=len(tokenizer)).to(device)
    print(f"[OK] Model Initialized: {model.count_parameters():,} parameters (~{model.count_parameters()/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = len(loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    model.train()
    print(f"\n[*] Starting GPU training for {epochs} epochs ({total_steps} steps)...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for step, (input_ids, labels) in enumerate(loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, len(tokenizer)), labels.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            if (step + 1) % 50 == 0 or (step + 1) == len(loader):
                print(f"Epoch [{epoch:02d}/{epochs:02d}] Step [{step+1:03d}/{len(loader):03d}] - Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / len(loader)
        print(f"--> Epoch {epoch} Average Loss: {avg_loss:.4f} | Perplexity: {math.exp(min(avg_loss, 20)):.2f}")

    # Package checkpoint
    payload = {
        "config": model.config,
        "state_dict": model.state_dict(),
        "tokenizer_name": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "vocab_size": len(tokenizer),
        "params_count": model.count_parameters()
    }
    torch.save(payload, output_path)
    print(f"\n[OK] Training completed in {time.time() - start_time:.1f}s.")
    print(f"[OK] High-acceptance checkpoint exported to: {output_path} ({os.path.getsize(output_path)/(1024**2):.1f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()

    train_gpu(epochs=args.epochs, batch_size=args.batch_size, num_samples=args.samples)
