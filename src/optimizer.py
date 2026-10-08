"""Numerically stable loss and a student implementation of AdamW."""

from __future__ import annotations

from collections.abc import Iterable

from copy import deepcopy
import torch
from torch import Tensor
from torch.optim import Optimizer


def stable_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int | None = None,
) -> Tensor:
    """Mean cross-entropy using an explicit stable log-sum-exp computation."""
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            "targets must match all logits dimensions except the vocabulary dimension"
        )

    maximum_logits = logits.max(
        dim=-1,
        keepdim=True,
    ).values

    log_sum_exp = (
        maximum_logits.squeeze(-1)
        + torch.log(
            torch.exp(logits - maximum_logits).sum(dim=-1)
        )
    )

    if ignore_index is None:
        safe_targets = targets
        valid_positions = None
    else:
        valid_positions = targets != ignore_index
        safe_targets = targets.masked_fill(
            ~valid_positions,
            0,
        )

    target_logits = logits.gather(
        dim=-1,
        index=safe_targets.unsqueeze(-1),
    ).squeeze(-1)

    losses = log_sum_exp - target_logits

    if valid_positions is not None:
        losses = losses[valid_positions]

    return losses.mean()


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

    def load_state_dict(self, state_dict: dict) -> None:
        """Load an independent copy of serialized optimizer state."""
        super().load_state_dict(
            deepcopy(state_dict)
        )


    @torch.no_grad()
    def step(self, closure=None):
        """Apply one decoupled AdamW update and return an optional closure loss."""
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            learning_rate = group["lr"]
            beta1, beta2 = group["betas"]
            epsilon = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                gradient = parameter.grad

                if gradient.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients"
                    )

                state = self.state[parameter]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                if weight_decay != 0:
                    parameter.mul_(
                        1 - learning_rate * weight_decay
                    )

                exp_avg.mul_(beta1).add_(
                    gradient,
                    alpha=1 - beta1,
                )

                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient,
                    gradient,
                    value=1 - beta2,
                )

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                corrected_step_size = (
                    learning_rate / bias_correction1
                )

                denominator = (
                    exp_avg_sq.sqrt()
                    / bias_correction2 ** 0.5
                ).add_(epsilon)

                parameter.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-corrected_step_size,
                )

        return loss
