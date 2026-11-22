"""Multilayer Perceptron (ANN) built with TensorFlow / Keras.

TF-IDF matrices stay sparse (SciPy CSR). Each mini-batch is densified on demand by
:class:`SparseBatchSequence`, so memory use is ``batch_size x n_features x 4 bytes``
(about 3.8 MB for 64 x 15,000) instead of the whole dense matrix (> 1 GB).
"""
from __future__ import annotations

import io
from typing import Any, Dict

import numpy as np
from scipy import sparse
from tensorflow import keras
from tensorflow.keras import layers


def set_seed(seed: int = 42) -> None:
    keras.utils.set_random_seed(seed)


def build_model(input_dim: int, num_classes: int, cfg: Dict[str, Any]) -> keras.Model:
    """Input -> Dense(256, ReLU) -> Dropout(.4) -> Dense(128, ReLU) -> Dropout(.3) -> Dense(20, softmax)."""
    mcfg = cfg["model"]
    units = mcfg["hidden_units"]
    drops = mcfg["dropout"]
    model = keras.Sequential(name="tfidf_mlp_document_classifier")
    model.add(layers.Input(shape=(input_dim,), name="tfidf_input"))
    for i, (u, d) in enumerate(zip(units, drops), start=1):
        model.add(layers.Dense(u, activation="relu", name=f"hidden_{i}"))
        model.add(layers.Dropout(d, name=f"dropout_{i}"))
    model.add(layers.Dense(num_classes, activation="softmax", name="output"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=mcfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class SparseBatchSequence(keras.utils.Sequence):
    """Feeds a SciPy sparse matrix to Keras one dense mini-batch at a time."""

    def __init__(self, X: sparse.spmatrix, y: np.ndarray, batch_size: int = 64,
                 shuffle: bool = True, seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.X = X.tocsr()
        self.y = np.asarray(y)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(self.X.shape[0])
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(self.X.shape[0] / self.batch_size))

    def __getitem__(self, idx: int):
        sel = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.X[sel].toarray().astype("float32"), self.y[sel]

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)


def predict_proba_batched(model: keras.Model, X: sparse.spmatrix, batch_size: int = 256) -> np.ndarray:
    """Class probabilities for a sparse matrix, densifying one batch at a time."""
    X = X.tocsr()
    n = X.shape[0]
    out = []
    for start in range(0, n, batch_size):
        batch = X[start:start + batch_size].toarray().astype("float32")
        out.append(model.predict(batch, verbose=0))
    return np.vstack(out) if out else np.zeros((0, model.output_shape[-1]), dtype="float32")


def model_summary_text(model: keras.Model) -> str:
    buf = io.StringIO()
    model.summary(print_fn=lambda line, **_: buf.write(line + "\n"))
    return buf.getvalue()
