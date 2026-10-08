"""Plot report-ready training curves from a metrics JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: Path) -> list[dict]:
    """Load one JSON object per non-empty line."""
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not rows:
        raise ValueError(f"no metrics found in {path}")

    return rows


def moving_average(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    """Compute a trailing moving average."""
    if window <= 0:
        raise ValueError("window must be positive")

    if len(values) < window:
        return values.copy()

    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("runs/gqa/metrics.jsonl"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("runs/gqa/training_curves"),
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=50,
    )
    args = parser.parse_args()

    rows = load_metrics(args.metrics)

    steps = np.asarray(
        [row["step"] for row in rows],
        dtype=np.int64,
    )
    training_loss = np.asarray(
        [row["training_loss"] for row in rows],
        dtype=np.float64,
    )
    learning_rate = np.asarray(
        [row["learning_rate"] for row in rows],
        dtype=np.float64,
    )
    gradient_norm = np.asarray(
        [row["gradient_norm"] for row in rows],
        dtype=np.float64,
    )
    throughput = np.asarray(
        [row["chunk_tokens_per_second"] for row in rows],
        dtype=np.float64,
    )

    validation_rows = [
        row
        for row in rows
        if "validation_mean_nll" in row
    ]
    validation_steps = np.asarray(
        [row["step"] for row in validation_rows],
        dtype=np.int64,
    )
    validation_nll = np.asarray(
        [row["validation_mean_nll"] for row in validation_rows],
        dtype=np.float64,
    )

    window = args.smoothing_window
    smoothed_steps = (
        steps[window - 1:]
        if len(steps) >= window
        else steps
    )
    smoothed_loss = moving_average(
        training_loss,
        window,
    )
    smoothed_gradient_norm = moving_average(
        gradient_norm,
        window,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8),
    )

    loss_axis = axes[0, 0]
    loss_axis.plot(
        steps,
        training_loss,
        color="tab:blue",
        alpha=0.15,
        linewidth=0.7,
        label="Training loss (raw)",
    )
    loss_axis.plot(
        smoothed_steps,
        smoothed_loss,
        color="tab:blue",
        linewidth=1.8,
        label=f"Training loss ({window}-step mean)",
    )
    loss_axis.plot(
        validation_steps,
        validation_nll,
        color="tab:orange",
        marker="o",
        markersize=3,
        linewidth=1.4,
        label="Validation mean NLL",
    )
    loss_axis.set_title("Training and Validation Loss")
    loss_axis.set_xlabel("Optimizer step")
    loss_axis.set_ylabel("NLL")
    loss_axis.legend()

    learning_rate_axis = axes[0, 1]
    learning_rate_axis.plot(
        steps,
        learning_rate,
        color="tab:green",
        linewidth=1.5,
    )
    learning_rate_axis.set_title("Learning-Rate Schedule")
    learning_rate_axis.set_xlabel("Optimizer step")
    learning_rate_axis.set_ylabel("Learning rate")
    learning_rate_axis.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
    )

    gradient_axis = axes[1, 0]
    gradient_axis.plot(
        steps,
        gradient_norm,
        color="tab:red",
        alpha=0.15,
        linewidth=0.7,
        label="Gradient norm (raw)",
    )
    gradient_axis.plot(
        smoothed_steps,
        smoothed_gradient_norm,
        color="tab:red",
        linewidth=1.8,
        label=f"Gradient norm ({window}-step mean)",
    )
    gradient_axis.set_title("Pre-Clipping Gradient Norm")
    gradient_axis.set_xlabel("Optimizer step")
    gradient_axis.set_ylabel("L2 norm")
    gradient_axis.legend()

    throughput_axis = axes[1, 1]
    throughput_axis.plot(
        steps,
        throughput / 1000.0,
        color="tab:purple",
        linewidth=1.5,
    )
    throughput_axis.set_title("Training Throughput")
    throughput_axis.set_xlabel("Optimizer step")
    throughput_axis.set_ylabel("Thousands of tokens/s")

    for axis in axes.flat:
        axis.grid(
            visible=True,
            alpha=0.25,
        )

    figure.suptitle(
        "GQA TinyStories Training",
        fontsize=15,
    )
    figure.tight_layout(
        rect=(0, 0, 1, 0.96),
    )

    args.output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    svg_path = args.output_prefix.with_suffix(".svg")
    pdf_path = args.output_prefix.with_suffix(".pdf")

    figure.savefig(
        svg_path,
        bbox_inches="tight",
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"loaded metric rows: {len(rows)}")
    print(f"validation points: {len(validation_rows)}")
    print(f"wrote {svg_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()