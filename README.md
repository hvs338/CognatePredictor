# CognatePredictor

A **Siamese Convolutional Neural Network** that detects *cognates* (historically
related words) among Meso-American languages — primarily Zapotec, Chatino, and
related Oto-Manguean varieties. The approach follows
[Rama (2016), *Siamese Convolutional Networks for Cognate Identification*](https://aclanthology.org/C16-1097.pdf).

Each word's phonetic "technical representation" (`tech_rep`) is encoded as a
**25×25 phonological-feature pseudo-image**: every phoneme becomes a 25-dim
feature vector (voiced, labial, stop, fricative, high/mid/low, …) from
[`Encoding_Table.xlsx`](Encoding_Table.xlsx), and up to 25 phonemes are stacked
into one image. Twin CNNs embed two words; a Euclidean-distance + sigmoid head
outputs a cognate probability. Ground-truth labels are the `Phonological
Cognate` set IDs (1–19) in [`conjoined_hash_table.xlsx`](conjoined_hash_table.xlsx).

## What this version adds (rigor)

This repo was refactored from a notebook into a runnable, reproducible package:

- **Runs anywhere** — paths are resolved relative to the repo root (no more
  hardcoded `/Users/.../Desktop` paths).
- **Leak-free evaluation** — words are split into train/val/test *before* pairs
  are built, so no word (or its pairs) leaks across splits. Reported numbers are
  honest test-set metrics (precision/recall/F1/ROC-AUC), not inflated accuracy.
- **Data-driven threshold** — the decision threshold is chosen by maximizing F1
  on validation, replacing the original magic `0.82`.
- **Reproducible** — all randomness is seeded; the model, hyperparameters,
  threshold, label map, and class centroids are saved as artifacts.
- **Two prediction tools** — score a word pair, or classify a word into a
  cognate set (the original README's "prediction tool … still being worked on").
- **A model that actually trains.** The original head fed a single Euclidean-
  distance scalar into `Dense(1)`, which bottlenecks gradients — on a leak-free
  split it collapsed to a constant ~0.5 output (flat loss). The twin embeddings
  are now merged by their element-wise absolute difference (a full vector) into a
  small Dense head, with BatchNorm and L2-normalized embeddings. The model now
  learns (see metrics below).
- **Two correctness fixes** to the original encoder:
  - *Symbol→feature alignment.* The original `np.array(values)[1:]` shifted every
    phoneme onto the next phoneme's feature vector (e.g. `A` got `B`'s features)
    and dropped `qu`. Now each phoneme maps to its own vector.
  - *Digraph-aware tokenization.* The original `list(word)` split digraphs like
    `zh`/`ch`/`ny` into single characters, ignoring the digraph columns the
    author defined. Now tokenization greedily matches the longest known symbol.

> Because evaluation is now leak-free and self-pairs are excluded, headline
> metrics will look **lower than the old notebook's** — that is the leakage being
> removed, not a regression. The dataset is small (~425 labeled words across 19
> cognate sets), so treat metrics as indicative.

### Reference run (held-out test pairs)

Default hyperparameters, `--epochs 80`, seed 42:

| accuracy | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.81 | 0.76 | 0.83 | 0.80 | 0.89 |

Threshold (0.39) is selected automatically by `evaluate.py`. Your numbers should
match closely given the fixed seed.

## Setup

TensorFlow supports Python 3.10–3.12 (not 3.13/3.14). If you only have a newer
Python, create a 3.12 environment — e.g. with [uv](https://docs.astral.sh/uv/):

```powershell
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Or with standard tooling:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Run from the repo root (the package is `cognate_predictor`).

```powershell
# 1. Train (uses the notebook's tuned hyperparameters by default)
python -m cognate_predictor.train --epochs 100
#    ...or run a hyperparameter search first:
python -m cognate_predictor.train --epochs 100 --tune --max-trials 10

# 2. Evaluate on the held-out test set + pick the decision threshold
python -m cognate_predictor.evaluate

# 3a. Score a word pair  ->  P(cognate)
python -m cognate_predictor.predict --score-pair "hue+róxo quijxi" "nèza quijxi"

# 3b. Classify a word into a cognate set
python -m cognate_predictor.predict --classify "hue+róxo quijxi" --topk 3
```

Artifacts are written to `artifacts/`: `siamese_model.keras`,
`feature_extractor.keras`, `hparams.json`, `label_map.json`, `threshold.json`,
`label_centroids.npz`, `metrics.json`.

## Project layout

| Path | Purpose |
|---|---|
| `cognate_predictor/config.py` | Paths, shapes, seeds, default hyperparameters |
| `cognate_predictor/encoding.py` | Phoneme → feature-vector encoding (the two fixes above) |
| `cognate_predictor/data.py` | Loading, leak-free split, pair generation |
| `cognate_predictor/model.py` | Siamese architecture + `AbsoluteDifference` (L1-merge) layer |
| `cognate_predictor/tuning.py` | `keras_tuner` RandomSearch |
| `cognate_predictor/train.py` | Train + save artifacts |
| `cognate_predictor/evaluate.py` | Test metrics + threshold selection |
| `cognate_predictor/predict.py` | Score-pair / classify CLI |

The original notebook and `*.py` scripts are kept for reference but are no longer
the run path.

## Notes & possible future work

- Cognate sets `0` and `20` exist in the data but were excluded by the original
  design (`LABEL_CLASS_IDS` in `config.py`); widening this is a one-line change.
- A contrastive/triplet loss, or nearest-neighbor search over all ~70k unlabeled
  words, would be natural extensions of the current embedding model.
