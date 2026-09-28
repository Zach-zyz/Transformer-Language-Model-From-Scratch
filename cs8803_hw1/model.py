"""Modern decoder-only Transformer components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
import torch.nn.functional as F
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
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(mean_square + self.eps)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        # Register precomputed non-persistent cosine and sine tables here.
        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )

        dimension_indices = torch.arange(
            0,
            head_dim,
            2,
            dtype=torch.float32,
        )

        inverse_frequencies = theta ** (
            -dimension_indices / head_dim
        )

        angles = (
            positions[:, None]
            * inverse_frequencies[None, :]
        )

        self.register_buffer(
            "cos_table",
            angles.cos(),
            persistent=False,
        )
        self.register_buffer(
            "sin_table",
            angles.sin(),
            persistent=False,
        )

    def forward(self, x: Tensor, position_offset: int = 0) -> Tensor:
        """Apply RoPE to a tensor shaped ``(..., sequence, head_dim)``."""
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"Expected head dimension {self.head_dim}, "
                f"received {x.shape[-1]}"
            )

        if position_offset < 0:
            raise ValueError("position_offset must be non-negative")

        sequence_length = x.shape[-2]
        position_end = position_offset + sequence_length

        if position_end > self.max_seq_len:
            raise ValueError(
                "RoPE positions exceed the configured maximum sequence length"
            )

        cos = self.cos_table[position_offset:position_end].to(
            device=x.device,
            dtype=x.dtype,
        )
        sin = self.sin_table[position_offset:position_end].to(
            device=x.device,
            dtype=x.dtype,
        )

        even = x[..., 0::2]
        odd = x[..., 1::2]

        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos

        rotated = torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        )

        return rotated.flatten(-2)


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
        batch_size, query_length, _ = x.shape

        q = self.q_proj(x)
        new_k = self.k_proj(x)
        new_v = self.v_proj(x)

        q = q.view(
            batch_size,
            query_length,
            self.config.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        new_k = new_k.view(
            batch_size,
            query_length,
            self.config.n_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        new_v = new_v.view(
            batch_size,
            query_length,
            self.config.n_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        past_length = 0

        if cache is not None:
            cached_k, cached_v = cache

            if cached_k.ndim != 4 or cached_v.ndim != 4:
                raise ValueError(
                    "KV cache tensors must be four-dimensional"
                )

            if cached_k.shape != cached_v.shape:
                raise ValueError(
                    "key and value caches must have identical shapes"
                )

            expected_prefix = (
                batch_size,
                self.config.n_kv_heads,
            )

            if cached_k.shape[:2] != expected_prefix:
                raise ValueError(
                    "KV cache batch or head dimensions do not match the input"
                )

            if cached_k.shape[-1] != self.head_dim:
                raise ValueError(
                    "KV cache head dimension does not match the model"
                )

            past_length = cached_k.shape[-2]

        q = self.rope(
            q,
            position_offset=past_length,
        )
        new_k = self.rope(
            new_k,
            position_offset=past_length,
        )

        if cache is None:
            compact_k = new_k
            compact_v = new_v
        else:
            cached_k, cached_v = cache

            compact_k = torch.cat(
                (cached_k, new_k),
                dim=-2,
            )
            compact_v = torch.cat(
                (cached_v, new_v),
                dim=-2,
            )

        updated_cache: KVCache | None = (
            (compact_k, compact_v)
            if use_cache
            else None
        )

        query_heads_per_kv_head = (
            self.config.n_heads
            // self.config.n_kv_heads
        )

        if query_heads_per_kv_head > 1:
            attention_k = compact_k.repeat_interleave(
                query_heads_per_kv_head,
                dim=1,
            )
            attention_v = compact_v.repeat_interleave(
                query_heads_per_kv_head,
                dim=1,
            )
        else:
            attention_k = compact_k
            attention_v = compact_v

        attention_scores = (
            q @ attention_k.transpose(-2, -1)
        )
        attention_scores = (
            attention_scores
            * (self.head_dim ** -0.5)
        )

        total_key_length = attention_k.shape[-2]

        key_positions = torch.arange(
            total_key_length,
            device=x.device,
        )
        query_positions = (
            past_length
            + torch.arange(
                query_length,
                device=x.device,
            )
        )

        causal_mask = (
            key_positions.unsqueeze(0)
            > query_positions.unsqueeze(1)
        )

        attention_scores = attention_scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )
        attention_weights = self.dropout(
            attention_weights
        )

        context = attention_weights @ attention_v

        context = (
            context
            .transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                query_length,
                self.config.d_model,
            )
        )

        output = self.o_proj(context)

        return output, updated_cache


class SwiGLU(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.w1(x))
        value = self.w3(x)
        hidden = gate * value
        return self.w2(hidden)


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
        attention_input = self.attn_norm(x)

        attention_output, new_cache = self.attn(
            attention_input,
            cache=cache,
            use_cache=use_cache,
        )

        x = x + attention_output

        ffn_input = self.ffn_norm(x)
        ffn_output = self.ffn(ffn_input)

        x = x + ffn_output

        return x, new_cache


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
        if isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )
            return

        if isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
            return

        if isinstance(module, nn.Linear):
            is_residual_projection = any(
                module is block.attn.o_proj
                or module is block.ffn.w2
                for block in self.blocks
            )

            if is_residual_projection:
                standard_deviation = (
                    0.02 / (2 * self.config.n_layers) ** 0.5
                )
            else:
                standard_deviation = 0.02

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=standard_deviation,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        cache: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        """Run full-sequence or incremental decoding with tied output weights."""
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape (batch, sequence)"
            )

        if cache is not None and len(cache) != len(self.blocks):
            raise ValueError(
                "cache must contain one KV cache per Transformer block"
            )

        sequence_length = input_ids.shape[1]

        if cache is None:
            past_length = 0
            layer_caches: list[KVCache | None] = [
                None
                for _ in self.blocks
            ]
        else:
            past_lengths = {
                layer_cache[0].shape[-2]
                for layer_cache in cache
            }

            if len(past_lengths) != 1:
                raise ValueError(
                    "all layer caches must have the same sequence length"
                )

            past_length = next(iter(past_lengths))
            layer_caches = list(cache)

        if past_length + sequence_length > self.config.max_seq_len:
            raise ValueError(
                "input sequence exceeds the configured maximum sequence length"
            )

        x = self.token_embedding(input_ids)

        new_cache: list[KVCache] | None = (
            [] if use_cache else None
        )

        for block, layer_cache in zip(
            self.blocks,
            layer_caches,
            strict=True,
        ):
            x, updated_layer_cache = block(
                x,
                cache=layer_cache,
                use_cache=use_cache,
            )

            if new_cache is not None:
                if updated_layer_cache is None:
                    raise RuntimeError(
                        "attention did not return a cache when use_cache=True"
                    )

                new_cache.append(updated_layer_cache)

        x = self.final_norm(x)

        logits = F.linear(
            x,
            self.token_embedding.weight,
        )

        return ModelOutput(
            logits=logits,
            cache=new_cache,
        )
