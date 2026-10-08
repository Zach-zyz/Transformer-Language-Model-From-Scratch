"""Run the end-to-end CPU/MPS smoke path."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generate import generate
from src.model import TransformerConfig, TransformerLM
from src.tokenizer import ByteBPETokenizer
from src.train import train_steps

def main() -> None:
    torch.manual_seed(2026)
    text = (ROOT / "data" / "toy_corpus.txt").read_text(encoding="utf-8")
    tokenizer = ByteBPETokenizer.train(text, vocab_size=512)
    token_ids = torch.tensor(tokenizer.encode(text, {tokenizer_special()}), dtype=torch.long)

    config_payload = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text())
    config = TransformerConfig(**config_payload)
    model = TransformerLM(config)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    result = train_steps(
        model,
        token_ids,
        steps=40,
        batch_size=4,
        sequence_length=32,
        learning_rate=3e-3,
        seed=2026,
        device=device,
    )
    if result.losses[-1] >= result.losses[0]:
        raise SystemExit("Smoke training did not reduce loss")

    model.eval()
    prompt_ids = torch.tensor([tokenizer.encode("The small model")], dtype=torch.long, device=device)
    output = generate(model, prompt_ids, max_new_tokens=12, temperature=0.0, use_cache=True)
    uncached_output = generate(
        model,
        prompt_ids,
        max_new_tokens=12,
        temperature=0.0,
        use_cache=False,
    )
    if not torch.equal(output, uncached_output):
        raise SystemExit("Cached and non-cached greedy generation differ")
    print(
        {
            "initial_loss": result.losses[0],
            "final_loss": result.losses[-1],
            "cached_matches_uncached": True,
            "sample": tokenizer.decode(output[0].cpu().tolist()),
        }
    )


def tokenizer_special() -> str:
    return "<|endoftext|>"


if __name__ == "__main__":
    main()
