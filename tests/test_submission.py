from __future__ import annotations

import pytest

from scripts.check_submission import validate_metrics


def valid_metrics() -> dict:
    cache_rows = [
        {
            "batch_size": batch_size,
            "analytical_cache_bytes": 1024,
            "cached_tokens_per_second": 100.0,
            "speedup": 1.1,
        }
        for batch_size in (1, 32)
    ]
    return {
        "gqa": {
            "parameter_count": 29_368_832,
            "completed_steps": 3000,
            "step_1000_validation": {
                "validation_mean_nll": 2.0,
                "validation_perplexity": 7.4,
            },
            "metrics": {
                "mean_nll": 1.8,
                "perplexity": 6.1,
                "bits_per_byte": 1.2,
                "target_tokens": 9160,
            },
            "cache_rows": cache_rows,
        },
        "mha": {
            "parameter_count": 30_155_776,
            "completed_steps": 1000,
            "step_1000_validation": {
                "validation_mean_nll": 2.1,
                "validation_perplexity": 8.2,
            },
            "cache_rows": cache_rows,
        },
    }


def test_submission_metrics_require_complete_experimental_evidence() -> None:
    with pytest.raises(ValueError, match="missing gqa"):
        validate_metrics({})
    incomplete = valid_metrics()
    incomplete["mha"]["cache_rows"] = incomplete["mha"]["cache_rows"][:1]
    with pytest.raises(ValueError, match="batch size 32"):
        validate_metrics(incomplete)


def test_submission_metrics_accept_report_summary_schema() -> None:
    validate_metrics(valid_metrics())
