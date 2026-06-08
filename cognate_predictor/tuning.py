"""Hyperparameter tuning via keras_tuner RandomSearch.

Modern replacement for the original ``kerastuner`` import. The search space
mirrors the original notebook; tuning runs on the train pairs and is validated
on the held-out validation pairs (not a random split of the train pairs).
"""

import keras_tuner as kt

from . import config
from .model import build_siamese_model

_HPARAM_KEYS = ["units1", "dropout1", "units2", "dropout2", "embedding_dim", "learning_rate"]


def _hypermodel(hp):
    params = {
        "units1": hp.Int("units1", 32, 512, step=16, default=128),
        "dropout1": hp.Float("dropout1", 0.0, 0.5, step=0.05, default=0.25),
        "units2": hp.Int("units2", 32, 512, step=32, default=128),
        "dropout2": hp.Float("dropout2", 0.0, 0.5, step=0.05, default=0.25),
        "embedding_dim": hp.Int("embedding_dim", 12, 60, step=12, default=48),
        "learning_rate": hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3),
    }
    model, _ = build_siamese_model(params)
    return model


def tune(train_pairs, val_pairs, max_trials, epochs, batch_size, seed):
    """Run RandomSearch and return the best hyperparameters as a plain dict.

    ``train_pairs`` / ``val_pairs`` are (left, right, labels) tuples.
    """
    xa, xb, y = train_pairs
    val_xa, val_xb, val_y = val_pairs

    tuner = kt.RandomSearch(
        _hypermodel,
        objective="val_accuracy",
        max_trials=max_trials,
        executions_per_trial=1,
        directory=str(config.TUNER_DIR),
        project_name="cognate",
        overwrite=True,
        seed=seed,
    )
    tuner.search(
        [xa, xb], y,
        validation_data=([val_xa, val_xb], val_y),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    best = tuner.get_best_hyperparameters()[0]
    return {k: best.get(k) for k in _HPARAM_KEYS}
