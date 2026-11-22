"""Train the TF-IDF + MLP document classifier.

Usage (from the project root)::

    python -m src.train                       # prepare data if needed, train, save artifacts
    python -m src.train --compare-metadata    # also run the with/without-metadata comparison
    python -m src.train --epochs 5            # quick smoke run

The test split is never used here. Hyper-parameter / preprocessing comparisons are
scored on the validation split only.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from datetime import datetime
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .config import ensure_dirs, load_config, resolve
from .data_loader import load_processed_dataset, prepare_dataset, split_frames
from .preprocessing import clean_text, fit_vectorizer, save_vectorizer

LOGGER = logging.getLogger(__name__)


def _to_jsonable(history: Dict[str, Any]) -> Dict[str, list]:
    return {k: [float(x) for x in v] for k, v in history.items()}


def train_model(cfg, X_train, y_train, X_val, y_val, max_epochs: int, verbose: int = 2):
    """Train the MLP with early stopping on validation loss. Returns (model, history, seconds)."""
    from tensorflow import keras

    from .ann_model import SparseBatchSequence, build_model, set_seed

    seed = cfg["project"]["seed"]
    mcfg = cfg["model"]
    set_seed(seed)
    model = build_model(X_train.shape[1], cfg["data"]["expected_num_classes"], cfg)
    train_seq = SparseBatchSequence(X_train, y_train, mcfg["batch_size"], shuffle=True, seed=seed)
    val_seq = SparseBatchSequence(X_val, y_val, mcfg["batch_size"], shuffle=False)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=mcfg["early_stopping_patience"], restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=mcfg["reduce_lr_factor"], patience=mcfg["reduce_lr_patience"],
            min_lr=mcfg["min_lr"], verbose=1,
        ),
    ]
    start = time.time()
    hist = model.fit(train_seq, validation_data=val_seq, epochs=max_epochs, callbacks=callbacks, verbose=verbose)
    return model, _to_jsonable(hist.history), time.time() - start


def run_training(cfg: Dict[str, Any], max_epochs: Optional[int] = None) -> Dict[str, Any]:
    import tensorflow as tf

    from .ann_model import model_summary_text

    ensure_dirs(cfg)
    df = load_processed_dataset(cfg) if resolve(cfg, "processed_csv").exists() else prepare_dataset(cfg)
    parts = split_frames(df)
    tr, va = parts["train"], parts["validation"]
    LOGGER.info("Split sizes: train=%d validation=%d test=%d (test untouched)", len(tr), len(va), len(parts["test"]))

    # Label encoder (verified against the dataset's own category ids)
    id_to_name = df.drop_duplicates("category_id").set_index("category_id")["category"].sort_index()
    encoder = LabelEncoder().fit(id_to_name.tolist())
    if not (encoder.transform(df["category"]) == df["category_id"].to_numpy()).all():
        raise RuntimeError("LabelEncoder ids do not match dataset category ids.")

    # TF-IDF: fit on TRAINING documents only, keep sparse
    LOGGER.info("Fitting TF-IDF on training documents only...")
    vec = fit_vectorizer(tr["cleaned_text"], cfg["tfidf"])
    X_train, X_val = vec.transform(tr["cleaned_text"]), vec.transform(va["cleaned_text"])
    y_train, y_val = tr["category_id"].to_numpy(), va["category_id"].to_numpy()
    LOGGER.info("TF-IDF features: %d, sparse nnz(train)=%d, dtype=%s", X_train.shape[1], X_train.nnz, X_train.dtype)

    epochs = max_epochs or cfg["model"]["max_epochs"]
    model, history, seconds = train_model(cfg, X_train, y_train, X_val, y_val, epochs)

    best_epoch = int(np.argmin(history["val_loss"])) + 1
    models_dir = resolve(cfg, "models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(resolve(cfg, "model_file"))
    save_vectorizer(vec, resolve(cfg, "vectorizer_file"))
    joblib.dump(encoder, resolve(cfg, "label_encoder_file"))

    mapping = {int(i): n for i, n in id_to_name.items()}
    with open(resolve(cfg, "category_mapping_file"), "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2)
    with open(resolve(cfg, "preprocessing_config_file"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "tfidf": cfg["tfidf"],
                "text_cleaning": "lowercase; remove URLs/e-mails; keep [a-z0-9]; collapse whitespace",
                "metadata_removed_in_training": cfg["data"]["metadata_to_remove"],
                "strip_metadata_on_inference": cfg["preprocessing"]["strip_metadata_on_inference"],
                "n_features": int(X_train.shape[1]),
            },
            fh, indent=2,
        )

    history_path = resolve(cfg, "training_history_file")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    summary = model_summary_text(model)
    (resolve(cfg, "reports_dir") / "model_summary.txt").write_text(summary, encoding="utf-8")

    info = {
        "status": "trained",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "training_duration_seconds": round(seconds, 1),
        "epochs_run": len(history["loss"]),
        "max_epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": float(min(history["val_loss"])),
        "val_accuracy_at_best_epoch": float(history["val_accuracy"][best_epoch - 1]),
        "train_accuracy_at_best_epoch": float(history["accuracy"][best_epoch - 1]),
        "n_features": int(X_train.shape[1]),
        "n_classes": len(mapping),
        "n_train": int(len(tr)), "n_validation": int(len(va)), "n_test_not_used": int(len(parts["test"])),
        "test_set_used_for_training_or_tuning": False,
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
        "config_snapshot": {k: cfg[k] for k in ("project", "data", "tfidf", "model")},
    }
    with open(resolve(cfg, "training_config_file"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    LOGGER.info("Training finished in %.1fs; best epoch %d; val acc %.4f",
                seconds, best_epoch, info["val_accuracy_at_best_epoch"])
    return info


def compare_metadata(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compare classification WITH vs WITHOUT metadata removal on the validation split.

    Headers (e.g. ``Newsgroups:``, ``Subject:``, organisation names) and quoted
    text can leak the label. The test split is not used for this decision.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_processed_dataset(cfg)
    parts = split_frames(df)
    tr, va = parts["train"], parts["validation"]
    results = {}
    variants = {
        "with_metadata (raw text)": lambda d: d["document_text"].map(clean_text),
        "without_metadata (headers/footers/quotes removed)": lambda d: d["cleaned_text"],
    }
    for name, getter in variants.items():
        LOGGER.info("Ablation variant: %s", name)
        t_tr, t_va = getter(tr), getter(va)
        vec = fit_vectorizer(t_tr, cfg["tfidf"])
        model, hist, secs = train_model(
            cfg, vec.transform(t_tr), tr["category_id"].to_numpy(),
            vec.transform(t_va), va["category_id"].to_numpy(), cfg["ablation"]["epochs"], verbose=2,
        )
        best = int(np.argmin(hist["val_loss"]))
        results[name] = {
            "best_epoch": best + 1,
            "val_accuracy": hist["val_accuracy"][best],
            "val_loss": hist["val_loss"][best],
            "train_accuracy": hist["accuracy"][best],
            "seconds": round(secs, 1),
        }
    out = {"scored_on": "validation split (test split untouched)", "epochs_budget": cfg["ablation"]["epochs"], "results": results}
    path = resolve(cfg, "metrics_dir") / "metadata_comparison.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["With metadata", "Without metadata"]
    vals = [r["val_accuracy"] for r in results.values()]
    ax.bar(labels, vals, color=["#9fa8da", "#3949ab"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Effect of Header/Footer/Quote Removal (validation split)")
    fig.tight_layout()
    fig.savefig(resolve(cfg, "figures_dir") / "metadata_comparison.png", dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train the ANN document classifier.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs.")
    parser.add_argument("--compare-metadata", action="store_true", help="Run metadata with/without comparison after training.")
    parser.add_argument("--compare-only", action="store_true", help="Only run the metadata comparison.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not args.compare_only:
        run_training(cfg, args.epochs)
    if args.compare_metadata or args.compare_only:
        compare_metadata(cfg)


if __name__ == "__main__":
    main()
