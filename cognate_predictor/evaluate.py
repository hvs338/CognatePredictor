"""Evaluate the trained model on the held-out test split and select a threshold.

Usage:
    python -m cognate_predictor.evaluate

Reconstructs the same deterministic splits/pairs as training, picks the
decision threshold that maximizes F1 on the validation pairs (replacing the
original magic 0.82), then reports honest metrics on the untouched test pairs.
Writes threshold.json and metrics.json.
"""

import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config
from .data import load_labeled, make_pairs, split_indices
from .encoding import encode_words
from .utils import seed_everything
# Importing model registers the custom AbsoluteDifference layer for load_model.
from . import model as _model  # noqa: F401


def _select_threshold(y_true, probs):
    """Threshold in [0.05, 0.95] that maximizes F1 on the validation pairs."""
    grid = np.linspace(0.05, 0.95, 181)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def main():
    seed_everything(config.SEED)
    import keras

    print("[INFO] Loading data and model...")
    words, labels, _ = load_labeled()
    images = encode_words(words)
    _, val_idx, test_idx = split_indices(labels)

    xa_va, xb_va, y_va = make_pairs(images[val_idx], labels[val_idx], config.PAIR_SEED_VAL)
    xa_te, xb_te, y_te = make_pairs(images[test_idx], labels[test_idx], config.PAIR_SEED_TEST)

    model = keras.models.load_model(config.SIAMESE_MODEL_PATH)

    # --- Threshold selection on validation ---------------------------------
    p_va = model.predict([xa_va, xb_va], verbose=0).ravel()
    threshold, val_f1 = _select_threshold(y_va.astype(int), p_va)
    config.THRESHOLD_PATH.write_text(json.dumps({"threshold": threshold}, indent=2))
    print(f"[INFO] Selected threshold={threshold:.3f} (val F1={val_f1:.3f})")

    # --- Honest metrics on the untouched test pairs ------------------------
    p_te = model.predict([xa_te, xb_te], verbose=0).ravel()
    y_true = y_te.astype(int)
    y_pred = (p_te >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "n_test_pairs": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, p_te))

    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("\n===== Test-set metrics (leak-free) =====")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}" if isinstance(v, float) else f"  {k:12s}: {v}")
    print(f"\n[INFO] Wrote {config.METRICS_PATH.name} and {config.THRESHOLD_PATH.name}")


if __name__ == "__main__":
    main()
