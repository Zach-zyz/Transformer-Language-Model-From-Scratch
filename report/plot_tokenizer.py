"""Render the tokenizer compression plot used by the LaTeX report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "outputs" / "tokenizer" / "benchmark.json"
OUTPUT = ROOT / "tmp" / "pdfs" / "tokenizer_vocabulary.pdf"


def main() -> None:
    rows = json.loads(BENCHMARK.read_text(encoding="utf-8"))["rows"]
    vocabulary_sizes = [row["vocabulary_size"] for row in rows]
    bytes_per_token = [row["bytes_per_token"] for row in rows]

    figure, axis = plt.subplots(figsize=(6.4, 3.25))
    axis.plot(
        vocabulary_sizes,
        bytes_per_token,
        marker="o",
        linewidth=2,
        color="#166534",
    )
    for vocabulary_size, value in zip(
        vocabulary_sizes,
        bytes_per_token,
        strict=True,
    ):
        axis.annotate(
            f"{value:.3f}",
            (vocabulary_size, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axis.set_xlabel("Vocabulary size")
    axis.set_ylabel("Bytes per token")
    axis.set_xticks(vocabulary_sizes)
    axis.grid(alpha=0.25)
    figure.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
