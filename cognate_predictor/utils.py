"""Small shared helpers."""

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow/Keras for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Imported lazily so non-Keras utilities (encoding, data) stay light.
    import keras

    keras.utils.set_random_seed(seed)
