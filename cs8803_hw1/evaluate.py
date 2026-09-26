"""Language-model evaluation helpers."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .model import TransformerLM


@torch.no_grad()
def evaluate_token_ids(
    model: TransformerLM,
    token_ids: Tensor,
    *,
    sequence_length: int,
    raw_byte_count: int,
) -> dict[str, float]:
    """Score every target token exactly once and return NLL, PPL, and bits per byte."""
    raise NotImplementedError
