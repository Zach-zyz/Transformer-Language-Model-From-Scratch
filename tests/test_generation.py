import pytest
import torch

from cs8803_hw1.generate import generate, sample_next_token
from cs8803_hw1.model import TransformerConfig, TransformerLM


def test_temperature_zero_is_greedy() -> None:
    logits = torch.tensor([[1.0, 4.0, 2.0]])
    assert sample_next_token(logits, temperature=0.0).item() == 1


def test_top_k_one_is_greedy() -> None:
    logits = torch.tensor([[1.0, 4.0, 2.0]])
    generator = torch.Generator().manual_seed(7)
    assert sample_next_token(logits, top_k=1, generator=generator).item() == 1


def test_top_k_keeps_exactly_k_with_lower_id_tie_break() -> None:
    logits = torch.zeros(1, 4)
    generator = torch.Generator().manual_seed(9)
    samples = [
        sample_next_token(logits, top_k=1, generator=generator).item()
        for _ in range(50)
    ]
    assert set(samples) == {0}


def test_invalid_sampling_arguments() -> None:
    logits = torch.zeros(1, 3)
    with pytest.raises(ValueError):
        sample_next_token(logits, temperature=-1)
    with pytest.raises(ValueError):
        sample_next_token(logits, top_k=0)
    with pytest.raises(ValueError):
        sample_next_token(logits, top_p=1.5)
    with pytest.raises(ValueError):
        sample_next_token(logits, top_p=0.0)


def test_top_p_removes_tail_tokens() -> None:
    logits = torch.log(torch.tensor([[0.45, 0.35, 0.20]]))
    generator = torch.Generator().manual_seed(26)
    samples = [
        sample_next_token(logits, top_p=0.8, generator=generator).item() for _ in range(500)
    ]
    assert set(samples) == {0, 1}


def test_cached_and_uncached_greedy_generation_match() -> None:
    torch.manual_seed(8)
    config = TransformerConfig(
        vocab_size=32,
        max_seq_len=24,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
    )
    model = TransformerLM(config).eval()
    prompt = torch.tensor([[1, 2, 3, 4]])
    cached = generate(model, prompt, max_new_tokens=6, temperature=0.0, use_cache=True)
    uncached = generate(model, prompt, max_new_tokens=6, temperature=0.0, use_cache=False)
    assert torch.equal(cached, uncached)


@pytest.mark.parametrize("n_kv_heads", [2, 4])
def test_cached_generation_projects_only_new_tokens_after_prompt(
    n_kv_heads: int,
) -> None:
    torch.manual_seed(32)
    config = TransformerConfig(
        vocab_size=32,
        max_seq_len=24,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=n_kv_heads,
        d_ff=64,
    )
    model = TransformerLM(config).eval()
    observed_lengths = {
        "key": [[] for _ in model.blocks],
        "value": [[] for _ in model.blocks],
    }
    hooks = []
    for layer_index, block in enumerate(model.blocks):
        hooks.append(
            block.attn.k_proj.register_forward_pre_hook(
                lambda module, args, index=layer_index: observed_lengths["key"][index].append(
                    args[0].shape[1]
                )
            )
        )
        hooks.append(
            block.attn.v_proj.register_forward_pre_hook(
                lambda module, args, index=layer_index: observed_lengths["value"][index].append(
                    args[0].shape[1]
                )
            )
        )
    try:
        generate(
            model,
            torch.tensor([[1, 2, 3, 4]]),
            max_new_tokens=4,
            temperature=0.0,
            use_cache=True,
        )
    finally:
        for hook in hooks:
            hook.remove()
    expected = [[4, 1, 1, 1], [4, 1, 1, 1]]
    assert observed_lengths == {"key": expected, "value": expected}
