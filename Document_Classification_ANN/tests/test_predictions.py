"""Prediction tests - require the trained model; skipped when it does not exist."""
import numpy as np
import pytest

from src.config import load_config
from src.predict import DocumentClassifier, EmptyDocumentError, ModelNotReadyError, artifact_status

pytest.importorskip("tensorflow")
READY = artifact_status(load_config())["ready"]
needs_model = pytest.mark.skipif(not READY, reason="trained model artifacts not present - run python -m src.train")


@pytest.fixture(scope="module")
def clf():
    return DocumentClassifier()


@needs_model
def test_single_prediction_is_valid_distribution(clf):
    proba = clf.predict_proba(["NASA launched a new satellite into orbit around the earth"])
    assert proba.shape == (1, 20) and np.isclose(proba.sum(), 1.0, atol=1e-4) and (proba >= 0).all()


@needs_model
def test_top_k_is_sorted_and_consistent(clf):
    res = clf.predict("The hockey team won the playoff game in overtime last night", top_k=5)
    probs = [t["probability"] for t in res["top_k"]]
    assert len(probs) == 5 and probs == sorted(probs, reverse=True)
    assert res["predicted_category"] == res["top_k"][0]["category"] in clf.class_names


@needs_model
def test_batch_matches_single(clf):
    docs = ["Encryption keys protect private communication", "Baseball pitchers throw fastballs at the plate"]
    batch = clf.predict_many(docs)
    for d, b in zip(docs, batch):
        assert clf.predict(d)["predicted_category"] == b["predicted_category"]


@needs_model
def test_empty_document_handling(clf):
    with pytest.raises(EmptyDocumentError):
        clf.predict("   !!! ")
    out = clf.predict_many(["", "valid text about motorcycles and riding gear"])
    assert "error" in out[0] and "predicted_category" in out[1]


def test_untrained_state_raises_clear_error(monkeypatch):
    monkeypatch.setattr("src.predict.artifact_status",
                        lambda cfg=None: {"ready": False, "missing": ["model"], "trained": False, "present": {}})
    with pytest.raises(ModelNotReadyError):
        DocumentClassifier()
