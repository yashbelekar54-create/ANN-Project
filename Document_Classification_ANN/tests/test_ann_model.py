"""ANN architecture / sparse pipeline tests. Need TensorFlow but NOT the trained model."""
import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("tensorflow")

from src.ann_model import SparseBatchSequence, build_model, predict_proba_batched, set_seed  # noqa: E402
from src.config import load_config  # noqa: E402


def test_architecture_matches_specification():
    cfg = load_config()
    model = build_model(1000, 20, cfg)
    kinds = [type(l).__name__ for l in model.layers]
    assert kinds == ["Dense", "Dropout", "Dense", "Dropout", "Dense"]
    assert [model.layers[0].units, model.layers[2].units, model.layers[4].units] == [256, 128, 20]
    assert model.layers[1].rate == 0.4 and model.layers[3].rate == 0.3
    assert model.output_shape == (None, 20)


def test_sparse_sequence_batches_are_dense_float32():
    X = sparse.random(130, 50, density=0.1, format="csr", dtype=np.float32, random_state=0)
    seq = SparseBatchSequence(X, np.arange(130) % 20, batch_size=64, shuffle=False)
    assert len(seq) == 3
    xb, yb = seq[0]
    assert xb.shape == (64, 50) and xb.dtype == np.float32 and yb.shape == (64,)
    assert seq[2][0].shape == (2, 50)


def test_tiny_training_run_and_batched_prediction():
    cfg = load_config()
    set_seed(42)
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.random((120, 40)).astype("float32") * (rng.random((120, 40)) > 0.8))
    y = rng.integers(0, 20, 120)
    model = build_model(40, 20, cfg)
    model.fit(SparseBatchSequence(X, y, 32), epochs=1, verbose=0)
    proba = predict_proba_batched(model, X, batch_size=50)
    assert proba.shape == (120, 20) and np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
