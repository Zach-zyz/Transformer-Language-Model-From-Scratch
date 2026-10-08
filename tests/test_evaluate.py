import math

import torch
from torch import nn

from src.evaluate import evaluate_token_ids
from src.model import ModelOutput


class UniformModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor) -> ModelOutput:
        batch, sequence = input_ids.shape
        logits = self.anchor + torch.zeros(batch, sequence, self.vocab_size)
        return ModelOutput(logits=logits)


def test_evaluation_scores_each_target_exactly_once() -> None:
    model = UniformModel(vocab_size=4).eval()
    tokens = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    metrics = evaluate_token_ids(model, tokens, sequence_length=3, raw_byte_count=18)
    assert metrics["target_tokens"] == 9
    assert math.isclose(metrics["mean_nll"], math.log(4), rel_tol=1e-6)
    assert math.isclose(metrics["perplexity"], 4.0, rel_tol=1e-6)
    assert math.isclose(metrics["bits_per_byte"], 1.0, rel_tol=1e-6)
