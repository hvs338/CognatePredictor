"""Phonological encoding: turn a word's technical representation into a
(MAX_LEN, NUM_FEATURES, 1) pseudo-image of phoneme feature vectors.

Two deliberate fixes over the original ``encoder.py`` (both documented in the
README):

1. **Correct symbol->feature alignment.** The original built the lookup with
   ``np.array(values)[1:]``, which shifted every symbol onto the *next*
   symbol's feature vector (e.g. 'A' received 'B's features) and dropped 'qu'
   entirely. Here each table column maps to its own feature vector.
2. **Digraph-aware tokenization.** The original used ``list(word)``, which
   split digraphs such as ``zh``/``ch``/``ny`` into single characters and
   never used the digraph columns the author defined. Here we greedily match
   the longest symbol (2-char digraphs first, then 1-char).
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config


@lru_cache(maxsize=1)
def _load_table():
    """Return (encoding_dict, max_symbol_len). Cached after first load."""
    table = pd.read_excel(config.ENCODING_TABLE_PATH).head(config.NUM_FEATURES).fillna(0)
    symbols = list(table.columns[1:])  # drop the 'Features' column
    encoding_dict = {
        sym: table[sym].to_numpy(dtype=np.float32) for sym in symbols
    }
    max_symbol_len = max(len(s) for s in symbols)
    return encoding_dict, max_symbol_len


def _tokenize(word: str, encoding_dict: dict, max_symbol_len: int):
    """Greedy longest-match tokenization into known phoneme symbols.

    Characters with no phonological features (spaces, '-', '+', digits, ...)
    are skipped, exactly as the original ignored unknown characters.
    """
    tokens = []
    i = 0
    n = len(word)
    while i < n:
        matched = False
        for length in range(max_symbol_len, 0, -1):
            chunk = word[i : i + length]
            if chunk in encoding_dict:
                tokens.append(chunk)
                i += length
                matched = True
                break
        if not matched:
            i += 1  # unknown character, skip
    return tokens


def encode_words(words) -> np.ndarray:
    """Encode an iterable of words into shape (N, MAX_LEN, NUM_FEATURES, 1).

    Returns exactly one row per input (non-string or empty inputs become an
    all-zero image) so the output stays aligned with any parallel label array.
    Words longer than MAX_LEN phonemes are truncated (post).
    """
    encoding_dict, max_symbol_len = _load_table()
    out = np.zeros(
        (len(words), config.MAX_LEN, config.NUM_FEATURES), dtype=np.float32
    )
    for row, word in enumerate(words):
        if not isinstance(word, str):
            continue
        tokens = _tokenize(word, encoding_dict, max_symbol_len)[: config.MAX_LEN]
        for col, tok in enumerate(tokens):
            out[row, col] = encoding_dict[tok]
    return out[..., np.newaxis]
