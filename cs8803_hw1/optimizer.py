"""Numerically stable loss and a student implementation of AdamW."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer


def stable_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int | None = None,
) -> Tensor:
    """Mean cross-entropy using an explicit stable log-sum-exp computation."""
    raise NotImplementedError


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("betas must be in [0, 1)")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Apply one decoupled AdamW update and return an optional closure loss."""
        raise NotImplementedError
