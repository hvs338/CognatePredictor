"""Central configuration: paths, shapes, seeds, and default hyperparameters.

All paths are resolved relative to the repository root so the project runs
unchanged on any machine (the original code hardcoded macOS Desktop paths).
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
ENCODING_TABLE_PATH = REPO_ROOT / "Encoding_Table.xlsx"
DATA_PATH = REPO_ROOT / "conjoined_hash_table.xlsx"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

SIAMESE_MODEL_PATH = ARTIFACTS_DIR / "siamese_model.keras"
FEATURE_EXTRACTOR_PATH = ARTIFACTS_DIR / "feature_extractor.keras"
HPARAMS_PATH = ARTIFACTS_DIR / "hparams.json"
LABEL_MAP_PATH = ARTIFACTS_DIR / "label_map.json"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"
CENTROIDS_PATH = ARTIFACTS_DIR / "label_centroids.npz"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
TUNER_DIR = ARTIFACTS_DIR / "tuner"

# --- Data / model shape ----------------------------------------------------
# 25 phonological feature rows (Encoding_Table head(25)) x 25 phoneme slots.
NUM_FEATURES = 25
MAX_LEN = 25
IMG_SHAPE = (MAX_LEN, NUM_FEATURES, 1)

# Cognate-set classes used as labels (matches the original notebook: ids 1..19).
# Class 0 and 20 exist in the data but were excluded by the original design.
LABEL_CLASS_IDS = list(range(1, 20))

# --- Reproducibility -------------------------------------------------------
SEED = 42
# Independent, deterministic seeds for pair generation per split so that
# train.py and evaluate.py reconstruct identical val/test pairs.
PAIR_SEED_TRAIN = SEED
PAIR_SEED_VAL = SEED + 1
PAIR_SEED_TEST = SEED + 2

# Train / val / test fractions (split is done on *words* before pairing).
SPLIT_RATIOS = (0.70, 0.15, 0.15)

# --- Default hyperparameters (the notebook's tuned, known-good values) ------
DEFAULT_HPARAMS = {
    "units1": 400,
    "dropout1": 0.2,
    "units2": 480,
    "dropout2": 0.15,
    "embedding_dim": 48,
    "learning_rate": 0.000891368139372892,
}

# Fallback decision threshold before evaluate.py selects an F1-optimal one.
DEFAULT_THRESHOLD = 0.5
