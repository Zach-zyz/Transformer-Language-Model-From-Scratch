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
    if token_ids.ndim != 1:
        raise ValueError(
            "token_ids must be a one-dimensional tensor"
        )

    if token_ids.numel() < 2:
        raise ValueError(
            "token_ids must contain at least two tokens"
        )

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be positive"
        )

    if raw_byte_count <= 0:
        raise ValueError(
            "raw_byte_count must be positive"
        )

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = token_ids.device

    total_nll = 0.0
    target_count = 0

    was_training = model.training
    model.eval()

    try:
        number_of_targets = token_ids.numel() - 1

        for start in range(
            0,
            number_of_targets,
            sequence_length,
        ):
            end = min(
                start + sequence_length,
                number_of_targets,
            )

            input_ids = token_ids[start:end]
            targets = token_ids[start + 1:end + 1]

            input_ids = input_ids.to(
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)

            targets = targets.to(
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)

            output = model(input_ids)

            logits = output.logits.float()

            log_probabilities = torch.log_softmax(
                logits,
                dim=-1,
            )

            target_log_probabilities = (
                log_probabilities.gather(
                    dim=-1,
                    index=targets.unsqueeze(-1),
                ).squeeze(-1)
            )

            total_nll += (
                -target_log_probabilities.sum().item()
            )

            target_count += targets.numel()

    finally:
        model.train(was_training)

    mean_nll = total_nll / target_count
    perplexity = math.exp(mean_nll)

    bits_per_byte = (
        total_nll
        / (raw_byte_count * math.log(2.0))
    )

    return {
        "mean_nll": mean_nll,
        "perplexity": perplexity,
        "bits_per_byte": bits_per_byte,
        "target_tokens": float(target_count),
    }
