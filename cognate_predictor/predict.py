"""Prediction CLI with the two requested modes.

Score a word pair (are these two words cognates?):
    python -m cognate_predictor.predict --score-pair "hue+róxo quijxi" "nèza quijxi"

Classify a word into a cognate set (nearest centroid in embedding space):
    python -m cognate_predictor.predict --classify "hue+róxo quijxi" --topk 3

Both modes can also be called programmatically via score_pair() / classify().
"""

import argparse
import json

import numpy as np

from . import config
from .encoding import encode_words
# Importing model registers the EuclideanDistance layer for deserialization.
from . import model as _model  # noqa: F401


def _load_threshold():
    if config.THRESHOLD_PATH.exists():
        return json.loads(config.THRESHOLD_PATH.read_text())["threshold"]
    return config.DEFAULT_THRESHOLD


def _load_label_map():
    raw = json.loads(config.LABEL_MAP_PATH.read_text())
    return {int(k): int(v) for k, v in raw.items()}


def score_pair(word_a, word_b):
    """Return cognate probability (0-1) for two words."""
    import keras

    siamese = keras.models.load_model(config.SIAMESE_MODEL_PATH)
    a = encode_words([word_a])
    b = encode_words([word_b])
    return float(siamese.predict([a, b], verbose=0).ravel()[0])


def classify(word, topk=3):
    """Rank cognate sets for a word by embedding distance to class centroids.

    Returns a list of dicts: {cognate_set_id, distance, within_cutoff}, best first.
    """
    import keras

    feature_extractor = keras.models.load_model(config.FEATURE_EXTRACTOR_PATH)
    label_map = _load_label_map()
    data = np.load(config.CENTROIDS_PATH)
    centroids, class_idx, cutoff = data["centroids"], data["class_idx"], float(data["cutoff"])

    emb = feature_extractor.predict(encode_words([word]), verbose=0)[0]
    dists = np.linalg.norm(centroids - emb, axis=1)
    order = np.argsort(dists)[:topk]
    return [
        {
            "cognate_set_id": label_map[int(class_idx[i])],
            "distance": float(dists[i]),
            "within_cutoff": bool(dists[i] <= cutoff),
        }
        for i in order
    ]


def main():
    parser = argparse.ArgumentParser(description="Predict cognate relationships.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--score-pair", nargs=2, metavar=("WORD_A", "WORD_B"))
    group.add_argument("--classify", metavar="WORD")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    if args.score_pair:
        word_a, word_b = args.score_pair
        prob = score_pair(word_a, word_b)
        threshold = _load_threshold()
        verdict = "COGNATE" if prob >= threshold else "not cognate"
        print(f"P(cognate) = {prob:.4f}  (threshold {threshold:.3f})  ->  {verdict}")
        print(f"  A: {word_a!r}\n  B: {word_b!r}")
    else:
        results = classify(args.classify, topk=args.topk)
        print(f"Top {len(results)} cognate sets for {args.classify!r}:")
        for rank, r in enumerate(results, 1):
            flag = "" if r["within_cutoff"] else "  (beyond cutoff -> weak/none)"
            print(f"  {rank}. cognate set {r['cognate_set_id']:>3}  "
                  f"distance={r['distance']:.4f}{flag}")


if __name__ == "__main__":
    main()
