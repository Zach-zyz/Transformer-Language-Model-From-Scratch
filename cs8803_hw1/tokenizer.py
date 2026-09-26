"""Deterministic byte-level BPE tokenizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SPECIAL_TOKEN = "<|endoftext|>"
SPECIAL_TOKEN_ID = 256


@dataclass
class ByteBPETokenizer:
    """A byte-level BPE tokenizer with a single reserved document-boundary token."""

    vocab: dict[int, bytes]
    merges: list[tuple[int, int]]

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "ByteBPETokenizer":
        """Train deterministic BPE.

        IDs 0-255 are bytes, ID 256 is ``<|endoftext|>``, and merge IDs start at 257.
        Pair counts never cross the special-token document boundary. Frequency ties are resolved by
        the ascending pair of token IDs.
        """
        raise NotImplementedError

    @property
    def special_token_id(self) -> int:
        return SPECIAL_TOKEN_ID

    def encode(
        self,
        text: str,
        allowed_special: set[str] | None = None,
    ) -> list[int]:
        """Encode text and reject special-token text unless explicitly allowed."""
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to UTF-8 text."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        payload = {
            "vocab": {str(idx): value.hex() for idx, value in self.vocab.items()},
            "merges": [list(pair) for pair in self.merges],
            "special_token": SPECIAL_TOKEN,
            "special_token_id": SPECIAL_TOKEN_ID,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["special_token_id"] != SPECIAL_TOKEN_ID:
            raise ValueError("Unexpected special-token ID")
        vocab = {int(idx): bytes.fromhex(value) for idx, value in payload["vocab"].items()}
        merges = [tuple(pair) for pair in payload["merges"]]
        return cls(vocab=vocab, merges=merges)
