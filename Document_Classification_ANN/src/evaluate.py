"""Evaluate the trained ANN on the untouched 20 Newsgroups test split.

Usage::

    python -m src.evaluate
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, log_loss,
                             precision_recall_fscore_support)

from .config import ensure_dirs, load_config, resolve
from .data_loader import load_processed_dataset, split_frames

LOGGER = logging.getLogger(__name__)
ACCENT, ACCENT_LIGHT = "#3949ab", "#9fa8da"


def plot_confusion_matrix(cm: np.ndarray, names, path, normalized: bool) -> None:
    data = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) if normalized else cm
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(data, annot=True, fmt=".2f" if normalized else "d", cmap="Blues", xticklabels=names,
                yticklabels=names, annot_kws={"size": 6}, cbar_kws={"label": "Proportion of true class" if normalized else "Documents"}, ax=ax)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title("Confusion Matrix - Test Set" + (" (row-normalised)" if normalized else " (counts)"))
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(history: Dict[str, list], best_epoch: int, path) -> None:
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (tr, va), title, ylabel in (
        (axes[0], ("accuracy", "val_accuracy"), "Training vs Validation Accuracy", "Accuracy"),
        (axes[1], ("loss", "val_loss"), "Training vs Validation Loss", "Loss (sparse categorical cross-entropy)"),
    ):
        ax.plot(epochs, history[tr], marker="o", color=ACCENT, label="Training")
        ax.plot(epochs, history[va], marker="s", color="#e65100", label="Validation")
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"Best epoch ({best_epoch})")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_category(report_df: pd.DataFrame, path) -> None:
    df = report_df.sort_values("f1-score")
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(df))
    h = 0.27
    ax.barh(y - h, df["precision"], h, label="Precision", color=ACCENT)
    ax.barh(y, df["recall"], h, label="Recall", color=ACCENT_LIGHT)
    ax.barh(y + h, df["f1-score"], h, label="F1-score", color="#e65100")
    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title("Per-Category Precision, Recall and F1-score - Test Set")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_terms_per_category(vec, X_train, y_train, names, k: int = 10) -> Dict[str, list]:
    terms = np.array(vec.get_feature_names_out())
    out = {}
    for cid, name in enumerate(names):
        mean = np.asarray(X_train[y_train == cid].mean(axis=0)).ravel()
        out[name] = terms[np.argsort(mean)[::-1][:k]].tolist()
    return out


def evaluate(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from tensorflow import keras

    from .ann_model import predict_proba_batched

    cfg = cfg or load_config()
    ensure_dirs(cfg)
    figures, metrics_dir, reports = resolve(cfg, "figures_dir"), resolve(cfg, "metrics_dir"), resolve(cfg, "reports_dir")

    df = load_processed_dataset(cfg)
    parts = split_frames(df)
    test, train = parts["test"], parts["train"]
    vec = joblib.load(resolve(cfg, "vectorizer_file"))
    encoder = joblib.load(resolve(cfg, "label_encoder_file"))
    model = keras.models.load_model(resolve(cfg, "model_file"))
    names = list(encoder.classes_)

    X_test, y_true = vec.transform(test["cleaned_text"]), test["category_id"].to_numpy()
    proba = predict_proba_batched(model, X_test)
    y_pred = proba.argmax(axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics = {
        "evaluated_on": "original 20 Newsgroups test split (untouched)",
        "n_test_documents": int(len(test)),
        "accuracy": acc,
        "test_loss": float(log_loss(y_true, proba, labels=list(range(len(names))))),
        "macro_precision": float(p_m), "macro_recall": float(r_m), "macro_f1": float(f_m),
        "weighted_precision": float(p_w), "weighted_recall": float(r_w), "weighted_f1": float(f_w),
    }

    rep = classification_report(y_true, y_pred, target_names=names, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).T
    rep_df.to_csv(reports / "classification_report.csv")
    (reports / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=names, digits=4, zero_division=0), encoding="utf-8")
    cat_df = rep_df.loc[names]
    weakest = cat_df.sort_values("f1-score").head(5)["f1-score"].round(4).to_dict()
    metrics["weakest_categories_by_f1"] = weakest

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    pd.DataFrame(cm, index=names, columns=names).to_csv(metrics_dir / "confusion_matrix.csv")
    plot_confusion_matrix(cm, names, figures / "confusion_matrix.png", normalized=False)
    plot_confusion_matrix(cm, names, figures / "confusion_matrix_normalized.png", normalized=True)
    plot_per_category(cat_df, figures / "per_category_metrics.png")

    pairs = [(names[i], names[j], int(cm[i, j])) for i in range(len(names)) for j in range(len(names)) if i != j and cm[i, j] > 0]
    pairs = sorted(pairs, key=lambda t: -t[2])[:10]
    pd.DataFrame(pairs, columns=["true_category", "predicted_as", "count"]).to_csv(metrics_dir / "most_confused_pairs.csv", index=False)

    history_path = resolve(cfg, "training_history_file")
    info = {}
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        info = json.loads(resolve(cfg, "training_config_file").read_text(encoding="utf-8"))
        plot_training_curves(history, info["best_epoch"], figures / "training_curves.png")

    X_train, y_train = vec.transform(train["cleaned_text"]), train["category_id"].to_numpy()
    (metrics_dir / "top_terms_per_category.json").write_text(
        json.dumps(top_terms_per_category(vec, X_train, y_train, names), indent=2), encoding="utf-8")

    (metrics_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.Series(metrics).to_csv(metrics_dir / "evaluation_metrics.csv", header=["value"])

    lines = [
        "# Measured results (auto-generated by `python -m src.evaluate`)", "",
        f"Test documents: {metrics['n_test_documents']} (original 20 Newsgroups test split, never used for training or tuning)", "",
        "| Metric | Value |", "|---|---|",
    ]
    for k in ("accuracy", "test_loss", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1"):
        lines.append(f"| {k} | {metrics[k]:.4f} |")
    if info:
        lines += ["", f"Training: {info['epochs_run']} epochs run, best epoch {info['best_epoch']}, "
                      f"{info['training_duration_seconds']} s, TensorFlow {info['tensorflow_version']}."]
    lines += ["", "Weakest categories (F1): " + ", ".join(f"{k} ({v})" for k, v in weakest.items()),
              "", "Most confused pairs: see `results/metrics/most_confused_pairs.csv`."]
    (reports / "results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Test accuracy %.4f | macro F1 %.4f | weighted F1 %.4f", acc, f_m, f_w)
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate the trained ANN on the test split.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    metrics = evaluate(load_config(args.config))
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
