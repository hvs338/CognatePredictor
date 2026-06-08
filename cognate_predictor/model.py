"""Siamese CNN architecture.

Twin CNNs (2x Conv->BatchNorm->ReLU->Pool->Dropout) embed each word, the
embeddings are merged by their element-wise absolute difference, and a small
Dense head outputs a cognate probability.

Two changes over the original notebook, both needed to make the model actually
train on a leak-free split (the original collapsed to constant ~0.5 output):

- **L1-merge head instead of a single Euclidean-distance scalar.** Feeding only
  one scalar distance into ``Dense(1)`` bottlenecks gradient flow so the
  embeddings never separate. The element-wise absolute difference is a full
  vector, giving the classifier (and thus the embeddings) a rich gradient.
- **BatchNorm + L2-normalized embeddings**, which restore gradient flow and keep
  embeddings on the unit sphere (also used by the centroid-based classify mode).

The merge is a registered custom layer (``AbsoluteDifference``) rather than a
``Lambda``, so ``model.save`` / ``load_model`` round-trip cleanly.
"""

import keras
from keras import layers

from . import config


@keras.saving.register_keras_serializable(package="cognate_predictor")
class AbsoluteDifference(layers.Layer):
    """Element-wise |a - b| between two batched embedding tensors."""

    def call(self, inputs):
        feats_a, feats_b = inputs
        return keras.ops.abs(feats_a - feats_b)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def build_feature_extractor(hparams) -> keras.Model:
    """The shared twin sub-network mapping an image to an embedding vector.

    Uses Conv -> BatchNorm -> ReLU blocks and an L2-normalized embedding. The
    original (Conv+ReLU only, raw embedding) collapsed during training: with a
    leak-free split it produced constant ~0.5 outputs and flat loss. BatchNorm
    restores gradient flow and UnitNormalization keeps embeddings on the unit
    sphere so the Euclidean distance stays a meaningful, bounded signal.
    """
    inputs = keras.Input(config.IMG_SHAPE)

    x = layers.Conv2D(int(hparams["units1"]), (2, 2), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(float(hparams["dropout1"]))(x)

    x = layers.Conv2D(int(hparams["units2"]), (2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(float(hparams["dropout2"]))(x)

    pooled = layers.GlobalAveragePooling2D()(x)
    embedding = layers.Dense(int(hparams["embedding_dim"]))(pooled)
    outputs = layers.UnitNormalization()(embedding)
    return keras.Model(inputs, outputs, name="feature_extractor")


def build_siamese_model(hparams):
    """Build and compile the full Siamese model.

    Returns (siamese_model, feature_extractor). The feature extractor shares
    weights with the siamese model and is used for embedding-based classification.
    """
    feature_extractor = build_feature_extractor(hparams)

    img_a = keras.Input(shape=config.IMG_SHAPE)
    img_b = keras.Input(shape=config.IMG_SHAPE)
    feats_a = feature_extractor(img_a)
    feats_b = feature_extractor(img_b)

    merged = AbsoluteDifference()([feats_a, feats_b])
    x = layers.Dense(16, activation="relu")(merged)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=[img_a, img_b], outputs=outputs)

    optimizer = keras.optimizers.Adam(learning_rate=float(hparams["learning_rate"]))
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, feature_extractor
