"""Training interfaces for the student release."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import math
import random
import numpy as np
import time
import torch
from torch import Tensor
from torch.optim import Optimizer

from .model import TransformerLM
from .optimizer import AdamW, stable_cross_entropy


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
        if self.steps <= 0:
            raise ValueError("steps must be positive")

        if self.micro_batch_size <= 0:
            raise ValueError(
                "micro_batch_size must be positive"
            )

        if self.sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive"
            )

        if self.accumulation_steps <= 0:
            raise ValueError(
                "accumulation_steps must be positive"
            )

        if self.peak_lr <= 0:
            raise ValueError(
                "peak_lr must be positive"
            )

        if self.min_lr < 0:
            raise ValueError(
                "min_lr must be non-negative"
            )

        if self.min_lr > self.peak_lr:
            raise ValueError(
                "min_lr must not exceed peak_lr"
            )

        if (
            self.warmup_steps < 0
            or self.warmup_steps > self.steps
        ):
            raise ValueError(
                "warmup_steps must be between 0 and steps"
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay must be non-negative"
            )

        if len(self.betas) != 2:
            raise ValueError(
                "betas must contain exactly two values"
            )

        beta1, beta2 = self.betas

        if not 0 <= beta1 < 1:
            raise ValueError(
                "beta1 must be in [0, 1)"
            )

        if not 0 <= beta2 < 1:
            raise ValueError(
                "beta2 must be in [0, 1)"
            )

        if self.grad_clip <= 0:
            raise ValueError(
                "grad_clip must be positive"
            )

        if self.precision not in {
            "fp32",
            "bf16",
            "fp16",
        }:
            raise ValueError(
                "precision must be fp32, bf16, or fp16"
            )

        if self.seed < 0:
            raise ValueError(
                "seed must be non-negative"
            )


def cosine_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Linear warmup followed by cosine decay."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    if step < 0:
        raise ValueError("step must be non-negative")

    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError(
            "warmup_steps must be between 0 and total_steps"
        )

    if peak_lr < 0 or min_lr < 0:
        raise ValueError(
            "learning rates must be non-negative"
        )

    if min_lr > peak_lr:
        raise ValueError(
            "min_lr must not exceed peak_lr"
        )

    if step >= total_steps:
        return min_lr

    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * step / warmup_steps

    decay_progress = (
        (step - warmup_steps)
        / (total_steps - warmup_steps)
    )

    cosine_factor = 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )

    return (
        min_lr
        + (peak_lr - min_lr) * cosine_factor
    )


def build_optimizer(
    model: TransformerLM,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
) -> AdamW:
    """Build AdamW parameter groups with no decay on vectors."""
    if learning_rate < 0:
        raise ValueError(
            "learning_rate must be non-negative"
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay must be non-negative"
        )

    decay_parameters: list[Tensor] = []
    no_decay_parameters: list[Tensor] = []

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    parameter_groups = []

    if decay_parameters:
        parameter_groups.append(
            {
                "params": decay_parameters,
                "weight_decay": weight_decay,
            }
        )

    if no_decay_parameters:
        parameter_groups.append(
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            }
        )

    return AdamW(
        parameter_groups,
        lr=learning_rate,
        betas=betas,
    )


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
    if steps <= 0:
        raise ValueError("steps must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    if accumulation_steps <= 0:
        raise ValueError(
            "accumulation_steps must be positive"
        )

    if gradient_clip <= 0:
        raise ValueError(
            "gradient_clip must be positive"
        )

    if start_step < 0:
        raise ValueError(
            "start_step must be non-negative"
        )

    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError(
            "precision must be fp32, bf16, or fp16"
        )

    if precision == "fp16" and device.type != "cuda":
        raise ValueError(
            "fp16 training requires a CUDA device"
        )

    if tokens.ndim != 1:
        raise ValueError(
            "tokens must be a one-dimensional packed array"
        )

    if tokens.numel() <= sequence_length:
        raise ValueError(
            "token array is too short for the requested sequence length"
        )

    if sequence_length > model.config.max_seq_len:
        raise ValueError(
            "sequence_length exceeds the model maximum"
        )

    if schedule_total_steps is None:
        schedule_total_steps = start_step + steps

    if schedule_total_steps < start_step + steps:
        raise ValueError(
            "schedule_total_steps is shorter than the requested run"
        )

    minimum_lr = (
        learning_rate
        if min_learning_rate is None
        else min_learning_rate
    )

    if minimum_lr < 0 or minimum_lr > learning_rate:
        raise ValueError(
            "min_learning_rate must be between 0 and learning_rate"
        )

    if warmup_steps < 0 or warmup_steps > schedule_total_steps:
        raise ValueError(
            "warmup_steps must be between 0 and schedule_total_steps"
        )

    model.to(device)
    model.train()

    if optimizer is None:
        optimizer = build_optimizer(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )

    if data_generator is None:
        data_generator = torch.Generator()
        data_generator.manual_seed(seed)

    use_scaler = (
        precision == "fp16"
        and device.type == "cuda"
    )

    active_scaler = scaler

    if use_scaler and active_scaler is None:
        active_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )

    if precision == "bf16":
        autocast_dtype = torch.bfloat16
    elif precision == "fp16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    started_at = time.perf_counter()

    losses: list[float] = []
    gradient_norms: list[float] = []
    learning_rates: list[float] = []

    window_length = sequence_length + 1
    maximum_start = tokens.numel() - sequence_length

    for local_step in range(steps):
        global_step = start_step + local_step

        current_lr = cosine_lr(
            global_step,
            schedule_total_steps,
            warmup_steps,
            learning_rate,
            minimum_lr,
        )

        for group in optimizer.param_groups:
            group["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)

        accumulated_loss: Tensor | None = None

        for _ in range(accumulation_steps):
            start_positions = torch.randint(
                low=0,
                high=maximum_start,
                size=(batch_size,),
                generator=data_generator,
            )

            offsets = torch.arange(
                window_length,
            )

            windows = tokens[
                start_positions[:, None]
                + offsets[None, :]
            ]

            input_ids = windows[:, :-1].to(
                device,
                non_blocking=True,
            )
            target_ids = windows[:, 1:].to(
                device,
                non_blocking=True,
            )

            if autocast_dtype is None:
                output = model(input_ids)
                loss = stable_cross_entropy(
                    output.logits,
                    target_ids,
                )
            else:
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                ):
                    output = model(input_ids)
                    loss = stable_cross_entropy(
                        output.logits,
                        target_ids,
                    )

            detached_loss = loss.detach()

            if accumulated_loss is None:
                accumulated_loss = detached_loss
            else:
                accumulated_loss = (
                    accumulated_loss + detached_loss
                )

            scaled_loss = (
                loss / accumulation_steps
            )

            if use_scaler:
                active_scaler.scale(
                    scaled_loss
                ).backward()
            else:
                scaled_loss.backward()

        if use_scaler:
            active_scaler.unscale_(optimizer)

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip,
        )

        if use_scaler:
            active_scaler.step(optimizer)
            active_scaler.update()
        else:
            optimizer.step()

        if accumulated_loss is None:
            raise RuntimeError(
                "no micro-batch loss was accumulated"
            )

        mean_loss = (
            accumulated_loss / accumulation_steps
        )

        losses.append(
            float(mean_loss.cpu())
        )
        gradient_norms.append(
            float(gradient_norm.detach().cpu())
        )
        learning_rates.append(
            float(current_lr)
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed_seconds = (
        time.perf_counter() - started_at
    )

    tokens_seen = (
        steps
        * batch_size
        * sequence_length
        * accumulation_steps
    )

    tokens_per_second = (
        tokens_seen / elapsed_seconds
    )

    if device.type == "cuda":
        peak_memory_bytes = int(
            torch.cuda.max_memory_allocated(device)
        )
    else:
        peak_memory_bytes = 0

    return TrainResult(
        losses=losses,
        gradient_norms=gradient_norms,
        learning_rates=learning_rates,
        optimizer_steps=steps,
        tokens_seen=tokens_seen,
        elapsed_seconds=elapsed_seconds,
        tokens_per_second=tokens_per_second,
        peak_memory_bytes=peak_memory_bytes,
    )


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
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "data_generator_state": data_generator.get_state(),
        "scaler": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
        "extra": dict(extra) if extra is not None else {},
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )


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
    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model"]
    )
    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    data_generator.set_state(
        checkpoint["data_generator_state"]
    )

    if scaler is not None and checkpoint["scaler"] is not None:
        scaler.load_state_dict(
            checkpoint["scaler"]
        )

    random.setstate(
        checkpoint["python_rng_state"]
    )
    np.random.set_state(
        checkpoint["numpy_rng_state"]
    )
    torch.set_rng_state(
        checkpoint["torch_rng_state"]
    )

    cuda_rng_state = checkpoint["cuda_rng_state"]

    if (
        cuda_rng_state is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            cuda_rng_state
        )

    step = int(checkpoint["step"])
    extra = dict(checkpoint.get("extra", {}))

    return step, extra
