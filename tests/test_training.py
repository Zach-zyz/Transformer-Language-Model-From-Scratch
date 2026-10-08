import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from src.model import TransformerConfig, TransformerLM
from src.optimizer import AdamW
from src.train import (
    TrainingConfig,
    build_optimizer,
    cosine_lr,
    load_training_checkpoint,
    save_training_checkpoint,
    train_steps,
)


def test_cosine_schedule_boundaries() -> None:
    assert cosine_lr(0, 100, 10, 1e-3, 1e-4) == 0.0
    assert math.isclose(cosine_lr(10, 100, 10, 1e-3, 1e-4), 1e-3)
    assert math.isclose(cosine_lr(100, 100, 10, 1e-3, 1e-4), 1e-4)


def test_training_config_effective_tokens_and_validation() -> None:
    config = TrainingConfig(
        steps=10,
        micro_batch_size=2,
        sequence_length=8,
        accumulation_steps=4,
        peak_lr=1e-3,
        min_lr=1e-4,
        warmup_steps=1,
    )
    config.validate()
    assert config.effective_tokens_per_step == 64


def test_accumulation_matches_larger_micro_batch() -> None:
    torch.manual_seed(27)
    config = TransformerConfig(
        vocab_size=16,
        max_seq_len=16,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
    )
    base = TransformerLM(config)
    large_batch = deepcopy(base)
    accumulated = deepcopy(base)
    tokens = torch.arange(256, dtype=torch.long) % config.vocab_size
    common = dict(
        tokens=tokens,
        steps=2,
        sequence_length=8,
        learning_rate=1e-3,
        seed=2026,
        device=torch.device("cpu"),
        weight_decay=0.0,
    )
    train_steps(large_batch, batch_size=4, accumulation_steps=1, **common)
    train_steps(accumulated, batch_size=2, accumulation_steps=2, **common)
    for expected, actual in zip(large_batch.parameters(), accumulated.parameters(), strict=True):
        assert torch.allclose(actual, expected, atol=2e-6)


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    torch.manual_seed(28)
    config = TransformerConfig(
        vocab_size=16,
        max_seq_len=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
    )
    initial = TransformerLM(config)
    uninterrupted = deepcopy(initial)
    split = deepcopy(initial)
    tokens = torch.arange(256, dtype=torch.long) % config.vocab_size
    uninterrupted_optimizer = build_optimizer(
        uninterrupted,
        learning_rate=1e-3,
        weight_decay=0.1,
    )
    uninterrupted_generator = torch.Generator().manual_seed(2026)
    uninterrupted_result = train_steps(
        uninterrupted,
        tokens,
        steps=4,
        batch_size=2,
        sequence_length=8,
        learning_rate=1e-3,
        seed=2026,
        device=torch.device("cpu"),
        weight_decay=0.1,
        optimizer=uninterrupted_optimizer,
        data_generator=uninterrupted_generator,
        schedule_total_steps=4,
    )

    split_optimizer = build_optimizer(split, learning_rate=1e-3, weight_decay=0.1)
    split_generator = torch.Generator().manual_seed(2026)
    first_half = train_steps(
        split,
        tokens,
        steps=2,
        batch_size=2,
        sequence_length=8,
        learning_rate=1e-3,
        seed=2026,
        device=torch.device("cpu"),
        weight_decay=0.1,
        optimizer=split_optimizer,
        data_generator=split_generator,
        schedule_total_steps=4,
    )
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        path,
        model=split,
        optimizer=split_optimizer,
        step=2,
        data_generator=split_generator,
        extra={"schedule_total_steps": 4},
    )

    restored = TransformerLM(config)
    restored_optimizer = build_optimizer(restored, learning_rate=1e-3, weight_decay=0.1)
    restored_generator = torch.Generator()
    step, extra = load_training_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        data_generator=restored_generator,
    )
    assert step == 2
    assert extra == {"schedule_total_steps": 4}
    second_half = train_steps(
        restored,
        tokens,
        steps=2,
        batch_size=2,
        sequence_length=8,
        learning_rate=1e-3,
        seed=2026,
        device=torch.device("cpu"),
        weight_decay=0.1,
        optimizer=restored_optimizer,
        data_generator=restored_generator,
        start_step=step,
        schedule_total_steps=4,
    )
    for expected, actual in zip(
        uninterrupted.parameters(),
        restored.parameters(),
        strict=True,
    ):
        assert torch.equal(actual, expected)
    assert first_half.losses + second_half.losses == uninterrupted_result.losses
    assert second_half.learning_rates == uninterrupted_result.learning_rates[2:]


def test_checkpoint_restores_rng_generator_and_scaler_state(tmp_path: Path) -> None:
    class FakeScaler:
        def __init__(self, scale: float) -> None:
            self.scale = scale

        def state_dict(self) -> dict[str, float]:
            return {"scale": self.scale}

        def load_state_dict(self, state: dict[str, float]) -> None:
            self.scale = state["scale"]

    torch.manual_seed(33)
    random.seed(33)
    np.random.seed(33)
    config = TransformerConfig(
        vocab_size=16,
        max_seq_len=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
    )
    model = TransformerLM(config)
    optimizer = build_optimizer(model, learning_rate=1e-3, weight_decay=0.1)
    data_generator = torch.Generator().manual_seed(3300)
    scaler = FakeScaler(128.0)
    path = tmp_path / "state.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=7,
        data_generator=data_generator,
        scaler=scaler,
    )

    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(3)
    expected_data = torch.randint(0, 100, (3,), generator=data_generator)
    expected_cuda = None
    if torch.cuda.is_available():
        expected_cuda = torch.rand(3, device="cuda").cpu()

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    data_generator.manual_seed(999)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)
    scaler.scale = 1.0

    step, extra = load_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        data_generator=data_generator,
        scaler=scaler,
    )
    assert step == 7
    assert extra == {}
    assert scaler.scale == 128.0
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)
    assert torch.equal(
        torch.randint(0, 100, (3,), generator=data_generator),
        expected_data,
    )
    if expected_cuda is not None:
        assert torch.equal(torch.rand(3, device="cuda").cpu(), expected_cuda)


def test_smoke_overfit() -> None:
    torch.manual_seed(6)
    config = TransformerConfig(
        vocab_size=16,
        max_seq_len=16,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
    )
    model = TransformerLM(config)
    motif = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    tokens = motif.repeat(32)
    result = train_steps(
        model,
        tokens,
        steps=30,
        batch_size=4,
        sequence_length=8,
        learning_rate=5e-3,
        seed=2026,
        device=torch.device("cpu"),
    )
    assert result.losses[-1] < result.losses[0] * 0.6
    assert len(result.gradient_norms) == 30
    assert len(result.learning_rates) == 30
    assert result.optimizer_steps == 30
    assert result.tokens_seen == 30 * 4 * 8
    assert result.elapsed_seconds > 0
    assert result.tokens_per_second > 0
    assert result.peak_memory_bytes == 0
