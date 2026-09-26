"""Provided utility functions; students do not need to modify this file."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch import Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_batch(
    tokens: Tensor,
    batch_size: int,
    sequence_length: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    if tokens.ndim != 1:
        raise ValueError("tokens must be one-dimensional")
    if len(tokens) <= sequence_length:
        raise ValueError("token array is too short")
    starts = torch.randint(
        0,
        len(tokens) - sequence_length,
        (batch_size,),
        generator=generator,
    )
    x = torch.stack([tokens[start : start + sequence_length] for start in starts])
    y = torch.stack([tokens[start + 1 : start + sequence_length + 1] for start in starts])
    return x, y


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
