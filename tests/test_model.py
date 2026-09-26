import torch
import torch.nn.functional as F

from cs8803_hw1.model import (
    CausalSelfAttention,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    TransformerBlock,
    TransformerConfig,
    TransformerLM,
)
from cs8803_hw1.optimizer import stable_cross_entropy


def tiny_config(**overrides) -> TransformerConfig:
    values = dict(
        vocab_size=64,
        max_seq_len=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=96,
        dropout=0.0,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def reference_rope(x: torch.Tensor, position_offset: int, theta: float = 10_000.0) -> torch.Tensor:
    head_dim = x.shape[-1]
    positions = torch.arange(
        position_offset,
        position_offset + x.shape[-2],
        dtype=x.dtype,
        device=x.device,
    )
    inv_freq = theta ** (
        -torch.arange(0, head_dim, 2, dtype=x.dtype, device=x.device) / head_dim
    )
    angles = positions[:, None] * inv_freq[None, :]
    cos = angles.cos()
    sin = angles.sin()
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


def test_rmsnorm_matches_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8)
    norm = RMSNorm(8, eps=1e-6)
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    assert torch.allclose(norm(x), expected, atol=1e-6)


def test_rope_relative_position_dot_product() -> None:
    torch.manual_seed(1)
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    x = torch.randn(2, 3, 4, 8)
    actual = rope(x, 3)
    expected = reference_rope(x, 3)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert not torch.allclose(actual, x)

    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)
    dot_a = (rope(q, 2) * rope(k, 5)).sum()
    dot_b = (rope(q, 7) * rope(k, 10)).sum()
    assert torch.allclose(dot_a, dot_b, atol=1e-5)


def test_attention_matches_sdpa_for_mha() -> None:
    torch.manual_seed(2)
    config = tiny_config(n_kv_heads=4)
    attention = CausalSelfAttention(config)
    x = torch.randn(2, 5, config.d_model)
    actual, _ = attention(x)

    batch, seq, _ = x.shape
    head_dim = config.d_model // config.n_heads
    q = attention.q_proj(x).view(batch, seq, config.n_heads, head_dim).transpose(1, 2)
    k = attention.k_proj(x).view(batch, seq, config.n_heads, head_dim).transpose(1, 2)
    v = attention.v_proj(x).view(batch, seq, config.n_heads, head_dim).transpose(1, 2)
    q = reference_rope(q, 0)
    k = reference_rope(k, 0)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    expected = expected.transpose(1, 2).contiguous().view(batch, seq, config.d_model)
    expected = attention.o_proj(expected)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_attention_matches_independent_gqa_reference() -> None:
    torch.manual_seed(21)
    config = tiny_config(n_kv_heads=2)
    attention = CausalSelfAttention(config)
    x = torch.randn(2, 6, config.d_model)
    actual, _ = attention(x)

    batch, seq, _ = x.shape
    head_dim = config.d_model // config.n_heads
    q = attention.q_proj(x).view(batch, seq, config.n_heads, head_dim).transpose(1, 2)
    k = attention.k_proj(x).view(batch, seq, config.n_kv_heads, head_dim).transpose(1, 2)
    v = attention.v_proj(x).view(batch, seq, config.n_kv_heads, head_dim).transpose(1, 2)
    q = reference_rope(q, 0)
    k = reference_rope(k, 0)
    groups = config.n_heads // config.n_kv_heads
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    expected = expected.transpose(1, 2).contiguous().view(batch, seq, config.d_model)
    expected = attention.o_proj(expected)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_cached_logits_match_full_forward_for_gqa() -> None:
    torch.manual_seed(3)
    model = TransformerLM(tiny_config()).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    full_logits = model(input_ids).logits

    cache = None
    pieces = []
    for index in range(input_ids.shape[1]):
        output = model(input_ids[:, index : index + 1], cache=cache, use_cache=True)
        cache = output.cache
        pieces.append(output.logits)
    cached_logits = torch.cat(pieces, dim=1)
    assert torch.allclose(cached_logits, full_logits, atol=2e-5)
    assert cache is not None
    for key, value in cache:
        assert key.shape == (2, model.config.n_kv_heads, 8, model.config.d_model // 4)
        assert value.shape == key.shape


def test_chunked_cache_matches_full_forward_and_preserves_prefix() -> None:
    torch.manual_seed(31)
    model = TransformerLM(tiny_config()).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 9))
    full_logits = model(input_ids).logits

    first = model(input_ids[:, :4], use_cache=True)
    assert first.cache is not None
    second = model(input_ids[:, 4:], cache=first.cache, use_cache=True)
    assert second.cache is not None
    chunked_logits = torch.cat((first.logits, second.logits), dim=1)
    assert torch.allclose(chunked_logits, full_logits, atol=2e-5)
    for old_cache, new_cache in zip(first.cache, second.cache, strict=True):
        for old_tensor, new_tensor in zip(old_cache, new_cache, strict=True):
            assert torch.equal(new_tensor[..., : old_tensor.shape[-2], :], old_tensor)


def test_fixed_seed_initial_logits_and_loss() -> None:
    torch.manual_seed(1234)
    config = TransformerConfig(
        vocab_size=16,
        max_seq_len=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
        dropout=0.0,
    )
    model = TransformerLM(config).eval()
    input_ids = torch.tensor([[0, 1, 2]])
    targets = torch.tensor([[1, 2, 3]])
    logits = model(input_ids).logits
    expected_prefix = torch.tensor(
        [
            0.25176862,
            0.06549722,
            0.03911776,
            -0.00493382,
            -0.10743506,
            -0.09986121,
            0.04811813,
            -0.04405170,
        ]
    )
    assert torch.allclose(logits[0, 0, :8], expected_prefix, atol=1e-6)
    assert torch.allclose(
        stable_cross_entropy(logits, targets),
        torch.tensor(2.82826805),
        atol=1e-6,
    )


def test_swiglu_matches_explicit_reference() -> None:
    torch.manual_seed(22)
    module = SwiGLU(tiny_config())
    x = torch.randn(2, 3, 32)
    expected = module.w2(F.silu(module.w1(x)) * module.w3(x))
    assert torch.allclose(module(x), expected, atol=1e-6)


def test_transformer_block_uses_pre_norm_residuals() -> None:
    class ScaleAttention(torch.nn.Module):
        def forward(self, x, cache=None, use_cache=False):
            del cache, use_cache
            return 2 * x, None

    class ScaleFFN(torch.nn.Module):
        def forward(self, x):
            return 3 * x

    block = TransformerBlock(tiny_config())
    block.attn_norm = torch.nn.Identity()
    block.ffn_norm = torch.nn.Identity()
    block.attn = ScaleAttention()
    block.ffn = ScaleFFN()
    x = torch.ones(1, 2, 32)
    actual, _ = block(x)
    assert torch.equal(actual, 12 * x)


def test_initialization_scales_residual_projections() -> None:
    torch.manual_seed(23)
    config = tiny_config(d_model=256, n_heads=8, n_kv_heads=4, d_ff=512, n_layers=4)
    model = TransformerLM(config)
    q_std = model.blocks[0].attn.q_proj.weight.std().item()
    o_std = model.blocks[0].attn.o_proj.weight.std().item()
    w2_std = model.blocks[0].ffn.w2.weight.std().item()
    expected_residual_std = 0.02 / (2 * config.n_layers) ** 0.5
    assert abs(q_std - 0.02) < 0.001
    assert abs(o_std - expected_residual_std) < 0.001
    assert abs(w2_std - expected_residual_std) < 0.001
    assert torch.equal(model.final_norm.weight, torch.ones_like(model.final_norm.weight))


def test_output_projection_is_tied_to_embedding() -> None:
    torch.manual_seed(24)
    model = TransformerLM(tiny_config()).train()
    input_ids = torch.tensor([[0, 1, 2]])
    output = model(input_ids).logits[..., 17].sum()
    output.backward()
    assert 17 not in input_ids
    assert model.token_embedding.weight.grad is not None
    assert model.token_embedding.weight.grad[17].abs().sum() > 0


def test_required_parameter_count() -> None:
    model = TransformerLM(TransformerConfig())
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == 29_368_832
