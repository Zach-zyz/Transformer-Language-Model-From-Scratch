"""Training interfaces for the student release."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor
from torch.optim import Optimizer

from .model import TransformerLM
from .optimizer import AdamW


@dataclass
class TrainResult:
    losses: list[float]
    gradient_norms: list[float]
    learning_rates: list[float]
    optimizer_steps: int
    tokens_seen: int
    elapsed_seconds: float
    tokens_per_second: float
    peak_memory_bytes: int


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration whose ``steps`` count optimizer updates, not micro-batches."""

    steps: int
    micro_batch_size: int
    sequence_length: int
    accumulation_steps: int
    peak_lr: float
    min_lr: float
    warmup_steps: int
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    seed: int = 2026

    @property
    def effective_tokens_per_step(self) -> int:
        return self.micro_batch_size * self.sequence_length * self.accumulation_steps

    def validate(self) -> None:
        raise NotImplementedError


def cosine_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Linear warmup followed by cosine decay."""
    raise NotImplementedError


def build_optimizer(
    model: TransformerLM,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
) -> AdamW:
    """Build AdamW parameter groups with no decay on vectors."""
    raise NotImplementedError


def train_steps(
    model: TransformerLM,
    tokens: Tensor,
    *,
    steps: int,
    batch_size: int,
    sequence_length: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    accumulation_steps: int = 1,
    warmup_steps: int = 0,
    min_learning_rate: float | None = None,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.95),
    gradient_clip: float = 1.0,
    precision: Literal["fp32", "bf16", "fp16"] = "fp32",
    optimizer: Optimizer | None = None,
    data_generator: torch.Generator | None = None,
    scaler: Any | None = None,
    start_step: int = 0,
    schedule_total_steps: int | None = None,
) -> TrainResult:
    """Run deterministic optimizer steps and support exact checkpoint continuation."""
    raise NotImplementedError


def save_training_checkpoint(
    path: str | Path,
    *,
    model: TransformerLM,
    optimizer: Optimizer,
    step: int,
    data_generator: torch.Generator,
    scaler: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save all state needed for an exact continuation."""
    raise NotImplementedError


def load_training_checkpoint(
    path: str | Path,
    *,
    model: TransformerLM,
    optimizer: Optimizer,
    data_generator: torch.Generator,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, dict[str, Any]]:
    """Restore an exact continuation and return ``(step, extra)``."""
    raise NotImplementedError
