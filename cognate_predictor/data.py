"""Data loading, leak-free splitting, and pair generation.

The core rigor fix lives here: words are split into train/val/test *before*
pairs are built, and pairs are generated separately within each split. This
guarantees no word (and none of its pairs) leaks across splits, unlike the
original which paired everything then applied ``validation_split``.
"""

import numpy as np
import pandas as pd

from . import config


def load_labeled():
    """Load labeled words and contiguous integer labels.

    Returns
    -------
    words : list[str]
    labels : np.ndarray[int]      contiguous 0..K-1
    label_map : dict[int, int]    contiguous index -> original cognate-set id
    """
    df = pd.read_excel(config.DATA_PATH)
    df["Phonological Cognate"] = df["Phonological Cognate"].fillna(-1)
    df = df[df["Phonological Cognate"].isin(config.LABEL_CLASS_IDS)].copy()
    df = df[df["tech_rep"].apply(lambda w: isinstance(w, str) and w.strip() != "")]

    words = df["tech_rep"].tolist()
    raw = df["Phonological Cognate"].astype(int).to_numpy()

    classes = sorted(np.unique(raw).tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    labels = np.array([class_to_idx[c] for c in raw], dtype=np.int64)
    label_map = {i: int(c) for i, c in enumerate(classes)}
    return words, labels, label_map


def split_indices(labels, ratios=config.SPLIT_RATIOS, seed=config.SEED):
    """Per-class (stratified) split of word indices into train/val/test.

    Splitting per class keeps every cognate set represented in train and, where
    it has enough members, in val/test too. Singleton classes go entirely to
    train. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = max(1, int(round(n * ratios[0])))
        rest = idx[n_train:]
        if len(rest) > 0:
            denom = ratios[1] + ratios[2]
            n_val = int(round(len(rest) * (ratios[1] / denom))) if denom > 0 else 0
            val.extend(rest[:n_val].tolist())
            test.extend(rest[n_val:].tolist())
        train.extend(idx[:n_train].tolist())
    return (
        np.array(sorted(train), dtype=np.int64),
        np.array(sorted(val), dtype=np.int64),
        np.array(sorted(test), dtype=np.int64),
    )


def make_pairs(images, labels, seed):
    """Build balanced positive/negative pairs within a single split.

    For each word: one positive pair with a *different* same-class word (skipped
    if the class is a singleton in this split, avoiding trivial self-pairs), and
    one negative pair with a random different-class word. Deterministic given
    ``seed``.

    Returns (left_images, right_images, pair_labels).
    """
    images = np.asarray(images)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    by_class = {c: np.where(labels == c)[0] for c in np.unique(labels)}

    left, right, pair_labels = [], [], []
    for i in range(len(images)):
        c = labels[i]
        same_other = by_class[c][by_class[c] != i]
        if len(same_other) > 0:
            j = rng.choice(same_other)
            left.append(images[i]); right.append(images[j]); pair_labels.append(1)
        diff = np.where(labels != c)[0]
        if len(diff) > 0:
            k = rng.choice(diff)
            left.append(images[i]); right.append(images[k]); pair_labels.append(0)

    return (
        np.asarray(left, dtype=np.float32),
        np.asarray(right, dtype=np.float32),
        np.asarray(pair_labels, dtype=np.float32),
    )
