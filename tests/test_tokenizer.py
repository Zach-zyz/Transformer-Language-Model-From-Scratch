from pathlib import Path
import random

import pytest

from src.tokenizer import ByteBPETokenizer, SPECIAL_TOKEN, SPECIAL_TOKEN_ID


def test_tokenizer_roundtrip_unicode_and_empty() -> None:
    tokenizer = ByteBPETokenizer.train("banana bandana", vocab_size=270)
    for text in ["", "hello", "caf\u00e9", "\u4f60\u597d", "\U0001f680 banana"]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_tie_break_and_reserved_special_id() -> None:
    tokenizer = ByteBPETokenizer.train("abac", vocab_size=258)
    assert tokenizer.special_token_id == SPECIAL_TOKEN_ID
    assert tokenizer.merges[0] == (ord("a"), ord("b"))
    assert set(range(258)) == set(tokenizer.vocab) | {SPECIAL_TOKEN_ID}


def test_learned_merges_are_used_by_encoder() -> None:
    tokenizer = ByteBPETokenizer.train("aaaa", vocab_size=258)
    assert tokenizer.merges == [(ord("a"), ord("a"))]
    assert tokenizer.encode("aaaa") == [257, 257]
    assert tokenizer.decode([257, 257]) == "aaaa"


def test_merge_priority_is_applied_in_training_order() -> None:
    vocab = {idx: bytes([idx]) for idx in range(256)}
    vocab[257] = b"ab"
    vocab[258] = b"abc"
    tokenizer = ByteBPETokenizer(vocab=vocab, merges=[(ord("a"), ord("b")), (257, ord("c"))])
    assert tokenizer.encode("abc") == [258]


def test_ranked_encoder_matches_sequential_merge_reference() -> None:
    random.seed(2026)
    training_text = "".join(random.choice("abcde ") for _ in range(400))
    tokenizer = ByteBPETokenizer.train(training_text, vocab_size=320)
    for _ in range(20):
        text = "".join(random.choice("abcde ") for _ in range(200))
        expected = list(text.encode("utf-8"))
        for merge_index, pair in enumerate(tokenizer.merges):
            merged = []
            index = 0
            while index < len(expected):
                if (
                    index + 1 < len(expected)
                    and expected[index] == pair[0]
                    and expected[index + 1] == pair[1]
                ):
                    merged.append(257 + merge_index)
                    index += 2
                else:
                    merged.append(expected[index])
                    index += 1
            expected = merged
        assert tokenizer.encode(text) == expected


def test_special_token_policy_and_document_boundary() -> None:
    tokenizer = ByteBPETokenizer.train(f"aa{SPECIAL_TOKEN}aa", vocab_size=258)
    with pytest.raises(ValueError):
        tokenizer.encode(f"a{SPECIAL_TOKEN}a")
    ids = tokenizer.encode(f"a{SPECIAL_TOKEN}a", {SPECIAL_TOKEN})
    assert SPECIAL_TOKEN_ID in ids
    assert tokenizer.decode(ids) == f"a{SPECIAL_TOKEN}a"
    repeated = tokenizer.encode(f"{SPECIAL_TOKEN}{SPECIAL_TOKEN}", {SPECIAL_TOKEN})
    assert repeated == [SPECIAL_TOKEN_ID, SPECIAL_TOKEN_ID]


def test_training_never_merges_across_document_boundary() -> None:
    tokenizer = ByteBPETokenizer.train(f"a{SPECIAL_TOKEN}b", vocab_size=258)
    assert tokenizer.merges == []


def test_save_load(tmp_path: Path) -> None:
    tokenizer = ByteBPETokenizer.train("the quick brown fox", vocab_size=265)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = ByteBPETokenizer.load(path)
    assert restored.vocab == tokenizer.vocab
    assert restored.merges == tokenizer.merges
    assert restored.encode("quick") == tokenizer.encode("quick")
