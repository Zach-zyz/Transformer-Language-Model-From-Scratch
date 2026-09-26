import torch
import torch.nn.functional as F

from cs8803_hw1.optimizer import AdamW, stable_cross_entropy


def test_stable_cross_entropy_matches_torch() -> None:
    torch.manual_seed(4)
    logits = torch.randn(3, 5, 11) * 20
    targets = torch.randint(0, 11, (3, 5))
    actual = stable_cross_entropy(logits, targets)
    expected = F.cross_entropy(logits.view(-1, 11), targets.view(-1))
    assert torch.allclose(actual, expected, atol=1e-6)


def test_stable_cross_entropy_handles_extreme_logits() -> None:
    logits = torch.tensor([[[10_000.0, -10_000.0, 0.0], [-10_000.0, 10_000.0, 0.0]]])
    targets = torch.tensor([[0, 1]])
    actual = stable_cross_entropy(logits, targets)
    assert torch.isfinite(actual)
    assert actual < 1e-6


def test_stable_cross_entropy_ignore_index() -> None:
    logits = torch.tensor([[[2.0, 0.0], [1.0, 3.0]]])
    targets = torch.tensor([[0, -1]])
    actual = stable_cross_entropy(logits, targets, ignore_index=-1)
    expected = F.cross_entropy(logits.view(-1, 2), targets.view(-1), ignore_index=-1)
    assert torch.allclose(actual, expected)


def test_adamw_matches_torch_for_multiple_steps() -> None:
    torch.manual_seed(5)
    ours_param = torch.nn.Parameter(torch.randn(4, 3))
    torch_param = torch.nn.Parameter(ours_param.detach().clone())
    ours = AdamW([ours_param], lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    reference = torch.optim.AdamW(
        [torch_param],
        lr=3e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    for step in range(5):
        gradient = torch.randn_like(ours_param) + step
        ours_param.grad = gradient.clone()
        torch_param.grad = gradient.clone()
        ours.step()
        reference.step()
        assert torch.allclose(ours_param, torch_param, atol=1e-7)


def test_adamw_parameter_groups_and_state_resume() -> None:
    torch.manual_seed(25)
    weight = torch.nn.Parameter(torch.randn(3, 3))
    norm = torch.nn.Parameter(torch.randn(3))
    optimizer = AdamW(
        [
            {"params": [weight], "weight_decay": 0.2},
            {"params": [norm], "weight_decay": 0.0},
        ],
        lr=1e-2,
        betas=(0.9, 0.95),
    )
    for _ in range(2):
        weight.grad = torch.randn_like(weight)
        norm.grad = torch.randn_like(norm)
        optimizer.step()

    resumed_weight = torch.nn.Parameter(weight.detach().clone())
    resumed_norm = torch.nn.Parameter(norm.detach().clone())
    resumed = AdamW(
        [
            {"params": [resumed_weight], "weight_decay": 0.2},
            {"params": [resumed_norm], "weight_decay": 0.0},
        ],
        lr=1e-2,
        betas=(0.9, 0.95),
    )
    resumed.load_state_dict(optimizer.state_dict())
    next_weight_grad = torch.randn_like(weight)
    next_norm_grad = torch.randn_like(norm)
    weight.grad = next_weight_grad.clone()
    norm.grad = next_norm_grad.clone()
    resumed_weight.grad = next_weight_grad.clone()
    resumed_norm.grad = next_norm_grad.clone()
    optimizer.step()
    resumed.step()
    assert torch.allclose(weight, resumed_weight, atol=1e-7)
    assert torch.allclose(norm, resumed_norm, atol=1e-7)
