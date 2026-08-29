"""
Custom Draft Transformer Decoder Architecture for Speculative Decoding.
Vocabulary size: 128,256 (Identical to Meta Llama 3.1).
Parameters: ~75M parameters with tied embedding & output LM head.
Designed for high-throughput, low-latency CPU forward passes.
"""

import math
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (matching Llama architecture)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) matching Llama 3.1."""
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

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if offset + seq_len > self.cos_cached.shape[0]:
            self._set_cos_sin_cache(offset + seq_len + 512)
        cos = self.cos_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, dim]
        sin = self.sin_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(1)
        return self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (self._rotate_half(x) * sin)

class SwiGLUMLP(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with KV-Cache support."""
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

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        offset = 0 if past_key_value is None else past_key_value[0].shape[2]
        q, k = self.rope(q, k, seq_len, offset=offset)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_kv = (k, v) if use_cache else None

        # Scaled dot-product attention with causal mask
        total_len = k.shape[2]
        if seq_len > 1:
            # Multi-token prompt phase
            attn_mask = torch.triu(torch.full((seq_len, total_len), float("-inf"), device=x.device), diagonal=1 + offset)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            # Single-token incremental step (attends to all past cached tokens)
            out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.o_proj(out), new_kv

class DecoderLayer(nn.Module):
    """Transformer Decoder Block."""
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, intermediate_dim: int, rope: RotaryEmbedding, norm_eps: float = 1e-5):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_dim, eps=norm_eps)
        self.self_attn = CausalSelfAttention(hidden_dim, num_heads, head_dim, rope)
        self.post_attention_layernorm = RMSNorm(hidden_dim, eps=norm_eps)
        self.mlp = SwiGLUMLP(hidden_dim, intermediate_dim)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        normed = self.input_layernorm(x)
        attn_out, new_kv = self.self_attn(normed, past_key_value=past_key_value, use_cache=use_cache)
        x = residual + attn_out

        residual = x
        normed = self.post_attention_layernorm(x)
        x = residual + self.mlp(normed)
        return x, new_kv

class DraftTransformerConfig:
    """Configuration class for the Draft Transformer."""
    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_dim: int = 512,
        num_layers: int = 5,
        num_heads: int = 8,
        head_dim: int = 64,
        intermediate_dim: int = 1536,
        max_seq_len: int = 4096,
        rope_base: float = 500000.0,
        norm_eps: float = 1e-5,
        tie_word_embeddings: bool = True
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.intermediate_dim = intermediate_dim
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        self.norm_eps = norm_eps
        self.tie_word_embeddings = tie_word_embeddings

    def to_dict(self) -> Dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: Dict) -> "DraftTransformerConfig":
        return cls(**d)

class DraftTransformer(nn.Module):
    """
    Speculative Draft Transformer Decoder.
    Shares identical 128,256 vocabulary with Llama 3.1.
    """
    def __init__(self, config: Optional[DraftTransformerConfig] = None):
        super().__init__()
        self.config = config or DraftTransformerConfig()

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_dim)
        self.rope = RotaryEmbedding(self.config.head_dim, max_seq_len=self.config.max_seq_len, base=self.config.rope_base)

        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                intermediate_dim=self.config.intermediate_dim,
                rope=self.rope,
                norm_eps=self.config.norm_eps
            ) for _ in range(self.config.num_layers)
        ])

        self.norm = RMSNorm(self.config.hidden_dim, eps=self.config.norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_dim, self.config.vocab_size, bias=False)

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        x = self.embed_tokens(input_ids)
        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, new_kv = layer(x, past_key_value=layer_past, use_cache=use_cache)
            if use_cache:
                new_past_key_values.append(new_kv)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_past_key_values

    @torch.no_grad()
    def propose_tokens(
        self,
        prefix_ids: torch.Tensor,
        num_draft_tokens: int = 4,
        temperature: float = 0.0
    ) -> List[int]:
        """
        Generates K candidate draft tokens given the current token prefix.
        Uses KV caching for fast sequential forward steps.
        """
        self.eval()
        device = prefix_ids.device
        if prefix_ids.dim() == 1:
            prefix_ids = prefix_ids.unsqueeze(0)

        # Prefill prompt prefix
        logits, past_kv = self.forward(prefix_ids, use_cache=True)
        next_logit = logits[:, -1, :]

        draft_tokens = []
        for _ in range(num_draft_tokens):
            if temperature > 0.0:
                probs = F.softmax(next_logit / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logit, dim=-1, keepdim=True)

            token_id = next_token.item()
            draft_tokens.append(token_id)

            # Fast step with cached KV
            logits, past_kv = self.forward(next_token, past_key_values=past_kv, use_cache=True)
            next_logit = logits[:, -1, :]

        return draft_tokens

if __name__ == "__main__":
    config = DraftTransformerConfig()
    model = DraftTransformer(config)
    params = model.count_parameters()
    print(f"DraftTransformer initialized successfully.")
    print(f"Total trainable parameters: {params:,} (~{params/1e6:.1f}M)")
