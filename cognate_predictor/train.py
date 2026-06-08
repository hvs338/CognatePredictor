"""Train the Siamese cognate model and save all prediction artifacts.

Usage:
    python -m cognate_predictor.train --epochs 100
    python -m cognate_predictor.train --epochs 100 --tune --max-trials 10

Saves to artifacts/: siamese_model.keras, feature_extractor.keras,
hparams.json, label_map.json, label_centroids.npz, and a default threshold.json
(refined later by evaluate.py).
"""

import argparse
import json

import numpy as np

from . import config
from .data import load_labeled, make_pairs, split_indices
from .encoding import encode_words
from .model import build_siamese_model
from .utils import seed_everything


def _compute_centroids(feature_extractor, train_images, train_labels):
    """Per-class embedding centroid + a 'none' distance cutoff for classify mode."""
    embeddings = feature_extractor.predict(train_images, verbose=0)
    class_idx = sorted(np.unique(train_labels).tolist())
    centroids = np.stack(
        [embeddings[train_labels == c].mean(axis=0) for c in class_idx]
    )
    # Cutoff: distance from each train embedding to its own centroid.
    pos = {c: i for i, c in enumerate(class_idx)}
    own_dists = np.array(
        [np.linalg.norm(embeddings[i] - centroids[pos[train_labels[i]]])
         for i in range(len(train_labels))]
    )
    cutoff = float(own_dists.mean() + 2.0 * own_dists.std())
    return centroids, np.array(class_idx, dtype=np.int64), cutoff


def main():
    parser = argparse.ArgumentParser(description="Train the Siamese cognate model.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tune", action="store_true", help="Run keras_tuner search first.")
    parser.add_argument("--max-trials", type=int, default=10)
    parser.add_argument("--tune-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    seed_everything(config.SEED)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data ---------------------------------------------------------------
    print("[INFO] Loading and encoding labeled data...")
    words, labels, label_map = load_labeled()
    images = encode_words(words)
    train_idx, val_idx, test_idx = split_indices(labels)
    print(f"[INFO] {len(words)} labeled words across {len(label_map)} cognate sets.")
    print(f"[INFO] Split (words): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_pairs = make_pairs(images[train_idx], labels[train_idx], config.PAIR_SEED_TRAIN)
    val_pairs = make_pairs(images[val_idx], labels[val_idx], config.PAIR_SEED_VAL)
    xa_tr, xb_tr, y_tr = train_pairs
    xa_va, xb_va, y_va = val_pairs
    print(f"[INFO] Pairs: train={len(y_tr)} val={len(y_va)}")

    # --- Hyperparameters ----------------------------------------------------
    if args.tune:
        from .tuning import tune
        print("[INFO] Tuning hyperparameters...")
        hparams = tune(train_pairs, val_pairs, args.max_trials, args.tune_epochs,
                       args.batch_size, config.SEED)
    else:
        hparams = dict(config.DEFAULT_HPARAMS)
    print(f"[INFO] Hyperparameters: {hparams}")

    # --- Train --------------------------------------------------------------
    import keras

    model, feature_extractor = build_siamese_model(hparams)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        )
    ]
    print("[INFO] Training...")
    model.fit(
        [xa_tr, xb_tr], y_tr,
        validation_data=([xa_va, xb_va], y_va),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # --- Save artifacts -----------------------------------------------------
    print("[INFO] Saving artifacts...")
    model.save(config.SIAMESE_MODEL_PATH)
    feature_extractor.save(config.FEATURE_EXTRACTOR_PATH)
    config.HPARAMS_PATH.write_text(json.dumps(hparams, indent=2))
    config.LABEL_MAP_PATH.write_text(
        json.dumps({str(k): v for k, v in label_map.items()}, indent=2)
    )

    centroids, class_idx, cutoff = _compute_centroids(
        feature_extractor, images[train_idx], labels[train_idx]
    )
    np.savez(config.CENTROIDS_PATH, centroids=centroids, class_idx=class_idx, cutoff=cutoff)

    if not config.THRESHOLD_PATH.exists():
        config.THRESHOLD_PATH.write_text(
            json.dumps({"threshold": config.DEFAULT_THRESHOLD}, indent=2)
        )

    print(f"[INFO] Done. Artifacts in {config.ARTIFACTS_DIR}")
    print("[INFO] Next: python -m cognate_predictor.evaluate")


if __name__ == "__main__":
    main()
