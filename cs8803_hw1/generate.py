"""Autoregressive sampling."""

from __future__ import annotations

import torch
from torch import Tensor

from .model import TransformerLM


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one token per batch item after temperature, top-k, and top-p filtering."""
    raise NotImplementedError


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt: Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int = 0,
    use_cache: bool = True,
) -> Tensor:
    """Generate tokens from a batched prompt."""
    raise NotImplementedError
