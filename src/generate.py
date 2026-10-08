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
    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape (batch, vocabulary)"
        )

    vocabulary_size = logits.shape[-1]

    if temperature < 0:
        raise ValueError(
            "temperature must be non-negative"
        )

    if top_k is not None:
        if top_k <= 0 or top_k > vocabulary_size:
            raise ValueError(
                "top_k must be between 1 and the vocabulary size"
            )

    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError(
                "top_p must be in the interval (0, 1]"
            )

    if temperature == 0:
        return logits.argmax(dim=-1)

    scaled_logits = logits / temperature

    sorted_logits, sorted_token_ids = torch.sort(
        scaled_logits,
        dim=-1,
        descending=True,
        stable=True,
    )

    if top_k is not None:
        sorted_logits = sorted_logits.clone()
        sorted_logits[:, top_k:] = float("-inf")

    if top_p is not None:
        sorted_probabilities = torch.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative_probabilities = (
            sorted_probabilities.cumsum(dim=-1)
        )

        probability_before_token = (
            cumulative_probabilities
            - sorted_probabilities
        )

        remove_mask = (
            probability_before_token >= top_p
        )

        sorted_logits = sorted_logits.masked_fill(
            remove_mask,
            float("-inf"),
        )

    probabilities = torch.softmax(
        sorted_logits,
        dim=-1,
    )

    sampled_sorted_positions = torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    )

    sampled_token_ids = sorted_token_ids.gather(
        dim=-1,
        index=sampled_sorted_positions,
    )

    return sampled_token_ids.squeeze(-1)


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
    if prompt.ndim != 2:
        raise ValueError(
            "prompt must have shape (batch, sequence)"
        )

    if prompt.shape[1] == 0:
        raise ValueError(
            "prompt must contain at least one token"
        )

    if max_new_tokens < 0:
        raise ValueError(
            "max_new_tokens must be non-negative"
        )

    total_length = prompt.shape[1] + max_new_tokens
    if total_length > model.config.max_seq_len:
        raise ValueError(
            "prompt and generated tokens exceed the maximum sequence length"
        )

    generated = prompt.clone()

    if max_new_tokens == 0:
        return generated

    generator = torch.Generator(
        device=generated.device
    ).manual_seed(seed)

    was_training = model.training
    model.eval()

    try:
        if use_cache:
            cache = None
            model_input = generated

            for _ in range(max_new_tokens):
                output = model(
                    model_input,
                    cache=cache,
                    use_cache=True,
                )

                cache = output.cache

                next_token = sample_next_token(
                    output.logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=generator,
                )

                generated = torch.cat(
                    [
                        generated,
                        next_token.unsqueeze(-1),
                    ],
                    dim=1,
                )

                model_input = next_token.unsqueeze(-1)

        else:
            for _ in range(max_new_tokens):
                output = model(
                    generated,
                    use_cache=False,
                )

                next_token = sample_next_token(
                    output.logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=generator,
                )

                generated = torch.cat(
                    [
                        generated,
                        next_token.unsqueeze(-1),
                    ],
                    dim=1,
                )

        return generated

    finally:
        model.train(was_training)
