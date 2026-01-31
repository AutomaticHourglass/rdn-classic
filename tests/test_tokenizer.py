# tests/test_tokenizer.py

import random
from pathlib import Path

# import pyo3_runtime  # contains PanicException raised by a Rust panic
import pytest

from rustbpe import Tokenizer

# Constants used in the tests
DOWN_ID = 256
UP_ID = 257

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]

# --------------------------------------------------------------------------- #
# Helper constants
DOWN_ID = 256
UP_ID = 257

SPECIAL_TOKENS_IDS = {
    "<|bos|>": 258,
    "<|user_start|>": 259,
    "<|user_end|>": 260,
    "<|assistant_start|>": 261,
    "<|assistant_end|>": 262,
    "<|python_start|>": 263,
    "<|python_end|>": 264,
    "<|output_start|>": 265,
    "<|output_end|>": 266,
}


# --------------------------------------------------------------------------- #
# Basic construction and defaults
def test_tokenizer_initialization():
    tokenizer = Tokenizer()
    assert isinstance(tokenizer.merges, dict)
    assert len(tokenizer.merges) == 0
    assert tokenizer.pattern == ""
    assert tokenizer.n_vocab == 0


# --------------------------------------------------------------------------- #
# Special‑token handling at the very beginning of a string
def test_encode_leading_special_token():
    tokenizer = Tokenizer()
    token_id = 258 + 3  # <|assistant_start|>
    encoded = tokenizer.encode("<|assistant_start|>Hello")
    assert encoded[0] == token_id


# --------------------------------------------------------------------------- #
# Round‑trip encode → decode (after training)
def test_encode_decode_roundtrip():
    tokenizer = Tokenizer()
    data = ["Hello world", "foo bar"]
    tokenizer.train_from_iterator(data, vocab_size=300)

    original = "Hello world"
    encoded = tokenizer.encode(original)
    decoded = tokenizer.decode(encoded)
    assert decoded == original


# --------------------------------------------------------------------------- #
def test_get_special_tokens():
    tokenizer = Tokenizer()
    assert tokenizer.get_special_tokens() == SPECIAL_TOKENS


# --------------------------------------------------------------------------- #
def test_get_vocab_size_after_training():
    tokenizer = Tokenizer()
    data = ["Hello world", "foo bar"]
    vocab_size = 300
    tokenizer.train_from_iterator(data, vocab_size=vocab_size)

    min_expected = 256 + 2 + 9  # base bytes + DOWN/UP + special tokens
    assert tokenizer.get_vocab_size() >= min_expected
    # The training may stop early if there are not enough distinct pairs.
    assert tokenizer.get_vocab_size() <= vocab_size


# --------------------------------------------------------------------------- #
def test_mergeable_ranks_length():
    tokenizer = Tokenizer()
    data = ["Hello world", "foo bar"]
    tokenizer.train_from_iterator(data, vocab_size=300)

    mergeable_ranks = tokenizer.get_mergeable_ranks()
    # Number of tokens is last_id + 1
    assert len(mergeable_ranks) == tokenizer.get_vocab_size() + 1


# --------------------------------------------------------------------------- #
def test_decode_each_token_matches_bytes():
    tokenizer = Tokenizer()
    data = ["Hello world", "foo bar"]
    tokenizer.train_from_iterator(data, vocab_size=300)

    mergeable_ranks = tokenizer.get_mergeable_ranks()
    for token_id, (bytes_seq, _) in enumerate(mergeable_ranks):
        # Skip recursion markers
        if token_id == DOWN_ID or token_id == UP_ID:
            continue

        try:
            expected_str = bytes_seq.decode("utf-8")
        except UnicodeDecodeError:
            # Skip tokens that are not valid UTF‑8
            continue

        decoded_str = tokenizer.decode([token_id])
        assert decoded_str == expected_str


# --------------------------------------------------------------------------- #
def test_tokenization_with_nested_delimiters():
    tokenizer = Tokenizer()
    text = "foo (bar [baz])"
    encoded = tokenizer.encode(text)

    expected_tokens = [
        DOWN_ID,
        102,
        111,
        111,
        UP_ID,
        32,
        40,
        DOWN_ID,
        98,
        97,
        114,
        UP_ID,
        32,
        91,
        DOWN_ID,
        98,
        97,
        122,
        UP_ID,
        93,
        41,
    ]

    assert encoded == expected_tokens


# # --------------------------------------------------------------------------- #
# def test_train_from_iterator_with_insufficient_vocab():
#     tokenizer = Tokenizer()
#     data = ["Hello world"]
#     with pytest.raises(PanicException):
#         tokenizer.train_from_iterator(data, vocab_size=200)


# --------------------------------------------------------------------------- #
def test_deterministic_merges():
    data = ["ab"] * 10
    vocab_size = 300

    tokenizer1 = Tokenizer()
    tokenizer2 = Tokenizer()

    tokenizer1.train_from_iterator(data, vocab_size=vocab_size)
    tokenizer2.train_from_iterator(data, vocab_size=vocab_size)

    assert tokenizer1.merges == tokenizer2.merges


# --------------------------------------------------------------------------- #
def test_merges_diff_with_different_vocab():
    random.seed(0)
    words_set = set()
    while len(words_set) < 200:
        w = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=5))
        words_set.add(w)
    data = list(words_set)

    tokenizer_small = Tokenizer()
    tokenizer_large = Tokenizer()

    tokenizer_small.train_from_iterator(data, vocab_size=300)
    tokenizer_large.train_from_iterator(data, vocab_size=400)

    # The larger vocabulary should produce at least as many merges
    assert len(tokenizer_small.merges) <= len(tokenizer_large.merges)


# --------------------------------------------------------------------------- #
def test_pattern_stored_correctly():
    tokenizer = Tokenizer()
    custom_pattern = r"\s+"
    data = ["Hello world"]
    tokenizer.train_from_iterator(data, vocab_size=300, pattern=custom_pattern)

    assert tokenizer.get_pattern() == custom_pattern


# --------------------------------------------------------------------------- #
def test_train_from_iterator_with_generator():
    tokenizer = Tokenizer()

    def generator():
        for i in range(5):
            yield f"sentence {i}"

    tokenizer.train_from_iterator(generator(), vocab_size=300)
    assert len(tokenizer.merges) > 0


# --------------------------------------------------------------------------- #
def test_encode_decode_empty_string():
    tokenizer = Tokenizer()
    encoded = tokenizer.encode("")
    assert encoded == []

    decoded = tokenizer.decode(encoded)
    assert decoded == ""


# --------------------------------------------------------------------------- #
def test_encode_word_after_merge():
    tokenizer = Tokenizer()
    data = ["ab"] * 100
    vocab_size = 270

    tokenizer.train_from_iterator(data, vocab_size=vocab_size)

    # Determine the new token id for the pair ('a','b')
    pair = (97, 98)  # 'a', 'b'
    expected_new_id = tokenizer.merges.get(pair)
    if expected_new_id is None:
        pytest.skip("Merge for 'ab' did not occur in training")

    encoded = tokenizer.encode("ab")
    # Expect: DOWN_ID, new_id, trailing UP_IDs are removed.
    assert len(encoded) == 2
    assert encoded[0] == DOWN_ID
    assert encoded[1] == expected_new_id


# --------------------------------------------------------------------------- #
def test_save_and_load(tmp_path: Path):
    tokenizer = Tokenizer()
    data = ["Hello world", "foo bar"]
    tokenizer.train_from_iterator(data, vocab_size=300)

    folder = tmp_path / "tokenizer_dir"
    tokenizer.save(str(folder))

    loaded = Tokenizer.load(str(folder))
    assert loaded.merges == tokenizer.merges
    assert loaded.pattern == tokenizer.pattern
    assert loaded.get_vocab_size() == tokenizer.get_vocab_size()


# --------------------------------------------------------------------------- #
def test_special_token_in_middle():
    """Special tokens that are *not* at the beginning should be treated as normal text."""
    tokenizer = Tokenizer()
    # Train with a dataset that contains the special token somewhere in the middle
    tokenizer.train_from_iterator(
        ["Hello world", "<|assistant_start|> hello"], vocab_size=300
    )

    text = "Hello <|assistant_start|> world"
    encoded = tokenizer.encode(text)

    # The special‑token id (261) should **not** appear in the output.
    assert all(tok != SPECIAL_TOKENS_IDS["<|assistant_start|>"] for tok in encoded)

    # Round‑trip must recover the original string.
    decoded = tokenizer.decode(encoded)
    assert decoded == text


# --------------------------------------------------------------------------- #
def test_nested_delimiters_mismatched():
    """A closing delimiter that does not match the most recent opener is treated as text."""
    tokenizer = Tokenizer()
    text = "foo (bar ]"
    encoded = tokenizer.encode(text)

    expected = [
        DOWN_ID,
        102,
        111,
        111,
        UP_ID,  # "foo"
        32,  # space
        40,  # '('
        DOWN_ID,
        98,
        97,
        114,
        UP_ID,  # "bar"
        32,  # space
        DOWN_ID,
        93,
    ]
    assert encoded == expected


# --------------------------------------------------------------------------- #
def test_trailing_space():
    """A trailing space should be emitted as a token after the final word."""
    tokenizer = Tokenizer()
    text = "foo bar "
    encoded = tokenizer.encode(text)

    # The last token must be the space character (32)
    assert encoded[-1] == 32


# --------------------------------------------------------------------------- #
def test_unicode_handling():
    """The tokenizer must preserve Unicode characters through encode/decode."""
    tokenizer = Tokenizer()
    data = ["café naïve"]
    tokenizer.train_from_iterator(data, vocab_size=300)

    text = "café naïve"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text


# --------------------------------------------------------------------------- #
def test_multiple_nested_delimiters():
    """Stack handling works for multiple nested delimiters."""
    tokenizer = Tokenizer()
    text = "((foo))"
    encoded = tokenizer.encode(text)

    expected = [
        40,
        40,  # '(' '('
        DOWN_ID,
        102,
        111,
        111,
        UP_ID,  # "foo"
        41,
        41,  # ')' ')'
    ]
    assert encoded == expected


# --------------------------------------------------------------------------- #
def test_all_delimiters():
    """All characters in DELIMITERS are emitted as raw byte tokens."""
    tokenizer = Tokenizer()
    text = " \t=:.,"  # space, tab, '=', ':', '.', ','
    encoded = tokenizer.encode(text)

    expected = [32, 9, 61, 58, 46, 44]
    assert encoded == expected


# --------------------------------------------------------------------------- #
def test_training_with_empty_iterator():
    """Training with an empty iterator should leave the tokenizer unchanged."""
    tokenizer = Tokenizer()
    empty_iter = iter([])
    tokenizer.train_from_iterator(empty_iter, vocab_size=300)

    assert len(tokenizer.merges) == 0
    # n_vocab remains zero – the tokenizer has not learned anything yet.
    assert tokenizer.n_vocab == 0


# --------------------------------------------------------------------------- #
def test_training_with_small_buffer():
    """Training should work correctly even with a very small buffer size."""
    tokenizer = Tokenizer()
    data = ["Hello world"] * 10
    tokenizer.train_from_iterator(data, vocab_size=300, buffer_size=1)

    assert len(tokenizer.merges) > 0


# --------------------------------------------------------------------------- #
def test_bpe_merging_of_long_word():
    """A long word should collapse into a single token after enough merges."""
    tokenizer = Tokenizer()
    data = ["aaaaaa"] * 10
    tokenizer.train_from_iterator(data, vocab_size=300)

    encoded = tokenizer.encode("aaaaaa")

    # The output should be a single merged token surrounded by markers.
    assert len(encoded) == 2
    assert encoded[0] == DOWN_ID
    # The inner token must not be the raw byte for 'a' (97).
    assert encoded[1] != 97


# --------------------------------------------------------------------------- #
def test_recursion_markers_not_in_output():
    """The only recursion markers that should appear are DOWN_ID and UP_ID."""
    tokenizer = Tokenizer()
    data = ["Hello world"]
    tokenizer.train_from_iterator(data, vocab_size=267)
    print(tokenizer.merges)

    encoded = tokenizer.encode("Hello world")

    # No other marker should be present in the output.
    print(encoded)
    assert all(tok in {256, 257} or (0 <= tok <= 255) for tok in encoded)


# --------------------------------------------------------------------------- #
def test_decode_skips_recursion_markers():
    """Decoding must ignore the recursion markers."""
    tokenizer = Tokenizer()
    data = ["Hello world"]
    tokenizer.train_from_iterator(data, vocab_size=300)

    token_ids = [256, 97, 98, 257]  # "ab" surrounded by markers
    decoded = tokenizer.decode(token_ids)

    assert decoded == "ab"


# --------------------------------------------------------------------------- #
def test_merge_id_range():
    """All merge ids must be unique and strictly increasing."""
    tokenizer = Tokenizer()
    data = ["ab"] * 10
    tokenizer.train_from_iterator(data, vocab_size=300)

    merge_ids = sorted(tokenizer.merges.values())
    assert len(merge_ids) == len(set(merge_ids))
    # They should be strictly increasing
    assert all(x < y for x, y in zip(merge_ids, merge_ids[1:]))


# --------------------------------------------------------------------------- #
def test_pattern_storage():
    """The tokenizer should remember the pattern that was passed during training."""
    custom_pattern = r"\s+"
    tokenizer = Tokenizer()
    data = ["Hello world"]
    tokenizer.train_from_iterator(data, vocab_size=300, pattern=custom_pattern)

    assert tokenizer.get_pattern() == custom_pattern


# --------------------------------------------------------------------------- #
def test_special_tokens_ids_correct():
    """The mapping from special token strings to ids must be consistent."""
    tokenizer = Tokenizer()
    data = ["Hello world"]
    tokenizer.train_from_iterator(data, vocab_size=300)

    for token_str, expected_id in SPECIAL_TOKENS_IDS.items():
        encoded = tokenizer.encode(token_str)
        # The first token should be the special‑token id
        assert encoded[0] == expected_id

    # Decoding should recover the original string.
    for token_str in SPECIAL_TOKENS_IDS:
        decoded = tokenizer.decode([SPECIAL_TOKENS_IDS[token_str]])
        assert decoded == token_str
