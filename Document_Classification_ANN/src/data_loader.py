"""Dataset acquisition, quality analysis and processed-CSV creation.

Usage (from the project root)::

    python -m src.data_loader            # download via scikit-learn + build processed CSV
    python -m src.data_loader --force    # rebuild even if the CSV already exists
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ensure_dirs, load_config, resolve
from .preprocessing import clean_text

LOGGER = logging.getLogger(__name__)


def _fetch(subset: str, remove: tuple, data_home: str):
    from sklearn.datasets import fetch_20newsgroups

    return fetch_20newsgroups(
        subset=subset, remove=remove, shuffle=False, data_home=data_home, download_if_missing=True
    )


def build_dataframe(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load the official 20 Newsgroups train/test sets through scikit-learn.

    ``document_text`` is the original, untouched post. ``text_no_metadata`` is the
    same post with headers, footers and quoted replies removed by scikit-learn.
    The original train/test separation is preserved in ``original_split``.
    """
    data_home = str(resolve(cfg, "raw_dir"))
    remove = tuple(cfg["data"]["metadata_to_remove"])
    frames = []
    for subset in ("train", "test"):
        full = _fetch(subset, (), data_home)
        stripped = _fetch(subset, remove, data_home)
        if list(full.target_names) != list(stripped.target_names) or (full.target != stripped.target).any():
            raise RuntimeError("Inconsistent labels between raw and stripped dataset loads.")
        names = list(full.target_names)
        frames.append(
            pd.DataFrame(
                {
                    "document_id": [f"{subset}_{i:05d}" for i in range(len(full.data))],
                    "source_file": [os.path.basename(p) for p in full.filenames],
                    "original_split": subset,
                    "document_text": full.data,
                    "text_no_metadata": stripped.data,
                    "category_id": full.target,
                    "category": [names[i] for i in full.target],
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)

    n_classes = df["category"].nunique()
    expected = cfg["data"]["expected_num_classes"]
    if n_classes != expected:
        raise RuntimeError(f"Expected {expected} categories but the dataset has {n_classes}.")
    return df


def add_splits(df: pd.DataFrame, val_size: float, seed: int) -> pd.DataFrame:
    """Create train / validation / test labels.

    The original test portion is kept untouched. The original training portion is
    split into train and validation with stratification.
    """
    df = df.copy()
    df["dataset_split"] = "test"
    train_idx = df.index[df["original_split"] == "train"]
    tr, va = train_test_split(
        train_idx, test_size=val_size, stratify=df.loc[train_idx, "category_id"], random_state=seed
    )
    df.loc[tr, "dataset_split"] = "train"
    df.loc[va, "dataset_split"] = "validation"
    return df


def quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Describe missing, empty and duplicate documents. Nothing is removed."""
    raw = df["document_text"]
    cleaned = df["cleaned_text"].fillna("")
    nonempty = df[cleaned.str.len() > 0]

    trainval_texts = set(nonempty.loc[nonempty["dataset_split"] != "test", "cleaned_text"])
    test_rows = nonempty[nonempty["dataset_split"] == "test"]
    train_texts = set(nonempty.loc[nonempty["dataset_split"] == "train", "cleaned_text"])
    val_rows = nonempty[nonempty["dataset_split"] == "validation"]

    word_counts = cleaned.str.split().str.len()
    report = {
        "total_documents": int(len(df)),
        "documents_per_split": df["dataset_split"].value_counts().to_dict(),
        "original_split_sizes": df["original_split"].value_counts().to_dict(),
        "num_categories": int(df["category"].nunique()),
        "category_counts_total": df["category"].value_counts().sort_index().to_dict(),
        "category_counts_by_split": df.groupby(["category", "dataset_split"]).size().unstack(fill_value=0).to_dict("index"),
        "missing_values": {c: int(df[c].isna().sum()) for c in ("document_text", "category", "category_id")},
        "empty_raw_documents": int((raw.fillna("").str.strip() == "").sum()),
        "empty_after_metadata_removal": int((df["text_no_metadata"].fillna("").str.strip() == "").sum()),
        "empty_after_cleaning": int((cleaned.str.len() == 0).sum()),
        "exact_duplicate_raw_documents": int(raw.duplicated().sum()),
        "duplicate_cleaned_documents_within_dataset": int(nonempty.duplicated("cleaned_text").sum()),
        "test_documents_whose_cleaned_text_also_in_train_or_validation": int(test_rows["cleaned_text"].isin(trainval_texts).sum()),
        "validation_documents_whose_cleaned_text_also_in_train": int(val_rows["cleaned_text"].isin(train_texts).sum()),
        "cleaned_word_count": {
            "min": int(word_counts.min()),
            "median": float(word_counts.median()),
            "mean": float(word_counts.mean()),
            "max": int(word_counts.max()),
        },
        "imbalance_ratio_max_to_min_class": float(df["category"].value_counts().max() / df["category"].value_counts().min()),
        "decisions": [
            "No document was removed from the dataset; empty or duplicate documents are only flagged.",
            "Empty-after-cleaning documents remain in the CSV (flag is_empty_after_cleaning); they yield an all-zero TF-IDF vector.",
            "The original train/test separation of 20 Newsgroups (by date) is preserved; the test split is never used for fitting or tuning.",
            "Validation data is a stratified 15% split of the original training portion.",
            "Primary text = scikit-learn 'remove=(headers, footers, quotes)' output, then clean_text().",
        ],
    }
    return report


def prepare_dataset(cfg: Optional[Dict[str, Any]] = None, force: bool = False) -> pd.DataFrame:
    """Download (if needed), clean, split and save the processed dataset CSV."""
    cfg = cfg or load_config()
    ensure_dirs(cfg)
    csv_path = resolve(cfg, "processed_csv")
    if csv_path.exists() and not force:
        LOGGER.info("Processed dataset already exists: %s", csv_path)
        return load_processed_dataset(cfg)

    LOGGER.info("Loading 20 Newsgroups through scikit-learn (downloads on first use)...")
    df = build_dataframe(cfg)
    df["cleaned_text"] = df["text_no_metadata"].map(clean_text)
    df["is_empty_after_cleaning"] = df["cleaned_text"].str.len() == 0
    df["is_duplicate_cleaned_text"] = df["cleaned_text"].duplicated(keep="first") & ~df["is_empty_after_cleaning"]
    df = add_splits(df, cfg["data"]["validation_size"], cfg["project"]["seed"])

    report = quality_report(df)
    out = resolve(cfg, "dataset_summary_file")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    ordered = [
        "document_id", "source_file", "original_split", "dataset_split", "category", "category_id",
        "document_text", "text_no_metadata", "cleaned_text", "is_empty_after_cleaning", "is_duplicate_cleaned_text",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df[ordered].to_csv(csv_path, index=False, encoding="utf-8")
    LOGGER.info("Saved %d documents to %s", len(df), csv_path)
    return df[ordered]


def load_processed_dataset(cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Load the processed CSV created by :func:`prepare_dataset`."""
    cfg = cfg or load_config()
    csv_path = resolve(cfg, "processed_csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run:  python -m src.data_loader")
    df = pd.read_csv(csv_path, keep_default_na=False)
    for col in ("document_text", "text_no_metadata", "cleaned_text"):
        df[col] = df[col].astype(str)
    return df


def split_frames(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {name: df[df["dataset_split"] == name] for name in ("train", "validation", "test")}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Download and prepare the 20 Newsgroups dataset.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true", help="Rebuild the processed CSV.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    df = prepare_dataset(cfg, force=args.force)
    print(df["dataset_split"].value_counts().to_string())
    print(f"Categories: {df['category'].nunique()}  Documents: {len(df)}")


if __name__ == "__main__":
    main()
