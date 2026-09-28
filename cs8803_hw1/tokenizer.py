"""Deterministic byte-level BPE tokenizer."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import heapq
from dataclasses import dataclass, field
from pathlib import Path


SPECIAL_TOKEN = "<|endoftext|>"
SPECIAL_TOKEN_ID = 256


@dataclass
class ByteBPETokenizer:
    """A byte-level BPE tokenizer with a single reserved document-boundary token."""

    vocab: dict[int, bytes]
    merges: list[tuple[int, int]]
    _merge_ranks: dict[tuple[int, int], int] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "ByteBPETokenizer":
        """Train deterministic BPE using local pair-count updates."""
        if vocab_size < 257:
            raise ValueError(
                "vocab_size must include 256 byte tokens and the special token"
            )

        vocab = {
            token_id: bytes([token_id])
            for token_id in range(256)
        }
        merges: list[tuple[int, int]] = []

        tokens: list[int] = []
        previous: list[int] = []
        following: list[int] = []
        alive: list[bool] = []

        for document in text.split(SPECIAL_TOKEN):
            document_tokens = list(document.encode("utf-8"))
            document_start = len(tokens)
            document_length = len(document_tokens)

            tokens.extend(document_tokens)

            previous.extend(
                -1 if offset == 0 else document_start + offset - 1
                for offset in range(document_length)
            )
            following.extend(
                -1
                if offset + 1 == document_length
                else document_start + offset + 1
                for offset in range(document_length)
            )
            alive.extend([True] * document_length)

        occurrences: defaultdict[tuple[int, int], set[int]] = defaultdict(set)

        for left_position, right_position in enumerate(following):
            if right_position != -1:
                pair = (
                    tokens[left_position],
                    tokens[right_position],
                )
                occurrences[pair].add(left_position)

        pair_heap = [
            (-len(positions), pair[0], pair[1])
            for pair, positions in occurrences.items()
        ]
        heapq.heapify(pair_heap)

        def remove_occurrence(
            pair: tuple[int, int],
            position: int,
            changed_pairs: set[tuple[int, int]],
        ) -> None:
            positions = occurrences.get(pair)

            if positions is not None and position in positions:
                positions.remove(position)
                changed_pairs.add(pair)

        def add_occurrence(
            pair: tuple[int, int],
            position: int,
            changed_pairs: set[tuple[int, int]],
        ) -> None:
            occurrences[pair].add(position)
            changed_pairs.add(pair)

        while len(vocab) + 1 < vocab_size:
            while pair_heap:
                negative_count, left_token_id, right_token_id = heapq.heappop(
                    pair_heap
                )
                best_pair = (left_token_id, right_token_id)
                current_count = len(occurrences.get(best_pair, ()))

                if current_count > 0 and -negative_count == current_count:
                    break
            else:
                break

            new_token_id = 257 + len(merges)
            vocab[new_token_id] = (
                vocab[left_token_id] + vocab[right_token_id]
            )
            merges.append(best_pair)

            changed_pairs: set[tuple[int, int]] = set()
            candidate_positions = sorted(occurrences[best_pair])

            for left_position in candidate_positions:
                if (
                    not alive[left_position]
                    or tokens[left_position] != left_token_id
                ):
                    continue

                right_position = following[left_position]

                if (
                    right_position == -1
                    or not alive[right_position]
                    or tokens[right_position] != right_token_id
                ):
                    continue

                previous_position = previous[left_position]
                next_position = following[right_position]

                if previous_position != -1:
                    old_left_pair = (
                        tokens[previous_position],
                        tokens[left_position],
                    )
                    remove_occurrence(
                        old_left_pair,
                        previous_position,
                        changed_pairs,
                    )

                remove_occurrence(
                    best_pair,
                    left_position,
                    changed_pairs,
                )

                if next_position != -1:
                    old_right_pair = (
                        tokens[right_position],
                        tokens[next_position],
                    )
                    remove_occurrence(
                        old_right_pair,
                        right_position,
                        changed_pairs,
                    )

                tokens[left_position] = new_token_id
                following[left_position] = next_position

                if next_position != -1:
                    previous[next_position] = left_position

                alive[right_position] = False
                previous[right_position] = -1
                following[right_position] = -1

                if previous_position != -1:
                    new_left_pair = (
                        tokens[previous_position],
                        new_token_id,
                    )
                    add_occurrence(
                        new_left_pair,
                        previous_position,
                        changed_pairs,
                    )

                if next_position != -1:
                    new_right_pair = (
                        new_token_id,
                        tokens[next_position],
                    )
                    add_occurrence(
                        new_right_pair,
                        left_position,
                        changed_pairs,
                    )

            for changed_pair in changed_pairs:
                count = len(occurrences[changed_pair])

                if count > 0:
                    heapq.heappush(
                        pair_heap,
                        (
                            -count,
                            changed_pair[0],
                            changed_pair[1],
                        ),
                    )

        return cls(vocab=vocab, merges=merges)

    @classmethod
    def _train_direct(cls, text: str, vocab_size: int) -> "ByteBPETokenizer":
        """Train deterministic BPE.

        IDs 0-255 are bytes, ID 256 is ``<|endoftext|>``, and merge IDs start at 257.
        Pair counts never cross the special-token document boundary. Frequency ties are resolved by
        the ascending pair of token IDs.
        """
        if vocab_size < 257:
            raise ValueError(
                "vocab_size must include 256 byte tokens and the special token"
            )

        vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        merges: list[tuple[int, int]] = []

        documents = [
            list(document.encode("utf-8"))
            for document in text.split(SPECIAL_TOKEN)
        ]

        while len(vocab) + 1 < vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()

            for document in documents:
                for index in range(len(document) - 1):
                    pair = (document[index], document[index + 1])
                    pair_counts[pair] += 1

            if not pair_counts:
                break

            best_pair = min(
                pair_counts,
                key=lambda pair: (
                    -pair_counts[pair],
                    pair[0],
                    pair[1],
                ),
            )

            new_token_id = 257 + len(merges)
            left_token_id, right_token_id = best_pair

            vocab[new_token_id] = (
                vocab[left_token_id] + vocab[right_token_id]
            )
            merges.append(best_pair)

            updated_documents: list[list[int]] = []

            for document in documents:
                updated_document: list[int] = []
                index = 0

                while index < len(document):
                    if (
                        index + 1 < len(document)
                        and document[index] == left_token_id
                        and document[index + 1] == right_token_id
                    ):
                        updated_document.append(new_token_id)
                        index += 2
                    else:
                        updated_document.append(document[index])
                        index += 1

                updated_documents.append(updated_document)

            documents = updated_documents

        return cls(vocab=vocab, merges=merges)

    @property
    def special_token_id(self) -> int:
        return SPECIAL_TOKEN_ID

    def encode(
        self,
        text: str,
        allowed_special: set[str] | None = None,
    ) -> list[int]:
        """Encode text using a ranked merge heap."""
        if allowed_special is None:
            allowed_special = set()

        if SPECIAL_TOKEN in text and SPECIAL_TOKEN not in allowed_special:
            raise ValueError(
                f"Special token {SPECIAL_TOKEN!r} is not allowed"
            )

        def encode_document(document: str) -> list[int]:
            tokens = list(document.encode("utf-8"))
            token_count = len(tokens)

            if token_count < 2:
                return tokens

            previous = [-1] + list(range(token_count - 1))
            following = list(range(1, token_count)) + [-1]
            alive = [True] * token_count

            pair_heap: list[tuple[int, int, int]] = []

            def push_pair(left_position: int) -> None:
                if left_position == -1 or not alive[left_position]:
                    return

                right_position = following[left_position]

                if right_position == -1 or not alive[right_position]:
                    return

                pair = (
                    tokens[left_position],
                    tokens[right_position],
                )
                rank = self._merge_ranks.get(pair)

                if rank is not None:
                    heapq.heappush(
                        pair_heap,
                        (
                            rank,
                            left_position,
                            right_position,
                        ),
                    )

            for left_position in range(token_count - 1):
                push_pair(left_position)

            while pair_heap:
                (
                    rank,
                    left_position,
                    expected_right_position,
                ) = heapq.heappop(pair_heap)

                if not alive[left_position]:
                    continue

                right_position = following[left_position]

                if (
                    right_position == -1
                    or right_position != expected_right_position
                    or not alive[right_position]
                ):
                    continue

                pair = (
                    tokens[left_position],
                    tokens[right_position],
                )

                if self._merge_ranks.get(pair) != rank:
                    continue

                previous_position = previous[left_position]
                next_position = following[right_position]

                tokens[left_position] = 257 + rank
                following[left_position] = next_position

                if next_position != -1:
                    previous[next_position] = left_position

                alive[right_position] = False
                previous[right_position] = -1
                following[right_position] = -1

                push_pair(previous_position)
                push_pair(left_position)

            encoded_document: list[int] = []
            position = 0

            while position != -1:
                encoded_document.append(tokens[position])
                position = following[position]

            return encoded_document

        encoded: list[int] = []
        documents = text.split(SPECIAL_TOKEN)

        for document_index, document in enumerate(documents):
            encoded.extend(encode_document(document))

            if document_index + 1 < len(documents):
                encoded.append(SPECIAL_TOKEN_ID)

        return encoded

    def _encode_direct(
        self,
        text: str,
        allowed_special: set[str] | None = None,
    ) -> list[int]:
        """Encode text and reject special-token text unless explicitly allowed."""
        if allowed_special is None:
            allowed_special = set()

        if SPECIAL_TOKEN in text and SPECIAL_TOKEN not in allowed_special:
            raise ValueError(
                f"Special token {SPECIAL_TOKEN!r} is not allowed"
            )

        def encode_document(document: str) -> list[int]:
            tokens = list(document.encode("utf-8"))

            for merge_index, (left_token_id, right_token_id) in enumerate(self.merges):
                new_token_id = 257 + merge_index
                merged_tokens: list[int] = []
                index = 0

                while index < len(tokens):
                    if (
                        index + 1 < len(tokens)
                        and tokens[index] == left_token_id
                        and tokens[index + 1] == right_token_id
                    ):
                        merged_tokens.append(new_token_id)
                        index += 2
                    else:
                        merged_tokens.append(tokens[index])
                        index += 1

                tokens = merged_tokens

            return tokens

        encoded: list[int] = []
        documents = text.split(SPECIAL_TOKEN)

        for document_index, document in enumerate(documents):
            encoded.extend(encode_document(document))

            if document_index + 1 < len(documents):
                encoded.append(SPECIAL_TOKEN_ID)

        return encoded

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to UTF-8 text."""
        decoded_bytes = bytearray()

        for token_id in token_ids:
            if token_id == SPECIAL_TOKEN_ID:
                decoded_bytes.extend(SPECIAL_TOKEN.encode("utf-8"))
            elif token_id in self.vocab:
                decoded_bytes.extend(self.vocab[token_id])
            else:
                raise ValueError(f"Unknown token ID: {token_id}")

        return bytes(decoded_bytes).decode("utf-8", errors="replace")

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
