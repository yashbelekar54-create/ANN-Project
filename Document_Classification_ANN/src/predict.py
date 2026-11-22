"""Inference: load the saved artifacts and classify new documents."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from .config import load_config, resolve
from .preprocessing import TextPreprocessor


class EmptyDocumentError(ValueError):
    """Raised when a document has no usable text."""


class ModelNotReadyError(RuntimeError):
    """Raised when the trained model or preprocessing artifacts are missing."""


def artifact_status(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Report which required artifacts exist and whether the model was actually trained."""
    cfg = cfg or load_config()
    keys = {"model": "model_file", "vectorizer": "vectorizer_file",
            "label_encoder": "label_encoder_file", "training_config": "training_config_file"}
    present = {name: resolve(cfg, key).exists() for name, key in keys.items()}
    trained = False
    if present["training_config"]:
        try:
            info = json.loads(resolve(cfg, "training_config_file").read_text(encoding="utf-8"))
            trained = info.get("status") == "trained"
        except (OSError, ValueError):
            trained = False
    return {"present": present, "trained": trained, "ready": all(present.values()) and trained,
            "missing": [n for n, ok in present.items() if not ok]}


class DocumentClassifier:
    """TF-IDF + trained ANN inference wrapper used by the application."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or load_config()
        status = artifact_status(self.cfg)
        if not status["ready"]:
            raise ModelNotReadyError(
                "Trained model artifacts are not available "
                f"(missing: {', '.join(status['missing']) or 'none'}; trained flag: {status['trained']}). "
                "Run  python -m src.train  first."
            )
        from tensorflow import keras  # imported lazily so tests/tools without TF can import this module

        self.model = keras.models.load_model(resolve(self.cfg, "model_file"))
        self.vectorizer = joblib.load(resolve(self.cfg, "vectorizer_file"))
        self.label_encoder = joblib.load(resolve(self.cfg, "label_encoder_file"))
        self.class_names: List[str] = list(self.label_encoder.classes_)
        self.preprocessor = TextPreprocessor(strip_meta=self.cfg["preprocessing"]["strip_metadata_on_inference"])
        self.n_features = len(self.vectorizer.vocabulary_)

    # ------------------------------------------------------------------ #
    def _transform(self, texts: List[str]):
        cleaned = [self.preprocessor(t) for t in texts]
        return cleaned, self.vectorizer.transform(cleaned)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        from .ann_model import predict_proba_batched

        _, X = self._transform(texts)
        return predict_proba_batched(self.model, X)

    def _format(self, proba: np.ndarray, known_terms: int, top_k: int) -> Dict[str, Any]:
        order = np.argsort(proba)[::-1][:top_k]
        top = [{"category": self.class_names[i], "probability": float(proba[i])} for i in order]
        result = {"predicted_category": top[0]["category"], "confidence": top[0]["probability"], "top_k": top,
                  "known_terms": int(known_terms), "warning": None}
        if known_terms == 0:
            result["warning"] = "None of the document's words are in the model vocabulary; the prediction is unreliable."
        return result

    def predict(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """Classify one document. Raises :class:`EmptyDocumentError` for empty text."""
        cleaned, X = self._transform([text])
        if not cleaned[0]:
            raise EmptyDocumentError("The document contains no usable text after cleaning.")
        from .ann_model import predict_proba_batched

        proba = predict_proba_batched(self.model, X)[0]
        return self._format(proba, X.nnz, top_k)

    def predict_many(self, texts: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Classify several documents; empty ones yield ``{'error': ...}`` entries."""
        from .ann_model import predict_proba_batched

        cleaned, X = self._transform(texts)
        valid = [i for i, c in enumerate(cleaned) if c]
        results: List[Dict[str, Any]] = [{"error": "The document contains no usable text after cleaning."} for _ in texts]
        if valid:
            proba = predict_proba_batched(self.model, X[valid])
            for row, i in enumerate(valid):
                results[i] = self._format(proba[row], X[i].nnz, top_k)
        return results
