"""Modern decoder-only Transformer components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import Tensor, nn


KVCache: TypeAlias = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 8192
    max_seq_len: int = 512
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    d_ff: int = 1536
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-6
    dropout: float = 0.0

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE head dimension must be even")


@dataclass
class ModelOutput:
    logits: Tensor
    cache: list[KVCache] | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        # Register precomputed non-persistent cosine and sine tables here.

    def forward(self, x: Tensor, position_offset: int = 0) -> Tensor:
        """Apply RoPE to a tensor shaped ``(..., sequence, head_dim)``."""
        raise NotImplementedError


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:
        """Return attention output and, when requested, the appended KV cache.

        Cache tensors use shape ``(batch, n_kv_heads, sequence, head_dim)``.
        """
        raise NotImplementedError


class SwiGLU(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_eps)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: Tensor,
        cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:
        raise NotImplementedError


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, config.rms_eps)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Apply the assignment initialization, including scaled residual projections."""
        raise NotImplementedError

    def forward(
        self,
        input_ids: Tensor,
        cache: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        """Run full-sequence or incremental decoding with tied output weights."""
        raise NotImplementedError
