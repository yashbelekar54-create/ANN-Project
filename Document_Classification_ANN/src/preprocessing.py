"""Text preprocessing and TF-IDF feature extraction.

The same functions are used for training, evaluation and for documents submitted
through the Streamlit application, so training and inference stay consistent.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

DEFAULT_TOKEN_PATTERN = r"(?u)\b[a-z0-9]{2,}\b"

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")
_HEADER_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\-]*:\s")
_ATTRIBUTION_RE = re.compile(r"^.*\b(writes|wrote|said)\s*:\s*$|^in article .*$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Basic text cleaning
# --------------------------------------------------------------------------- #
def clean_text(text: Any) -> str:
    """Lowercase, remove URLs / e-mails / punctuation and normalise whitespace.

    Letters and digits are kept because tokens such as ``486`` or ``x11`` carry
    topical information in this corpus.
    """
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _URL_RE.sub(" ", t)
    t = _EMAIL_RE.sub(" ", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def tokenize(text: str, token_pattern: str = DEFAULT_TOKEN_PATTERN) -> List[str]:
    """Split (already cleaned) text into tokens with the vectorizer's token pattern."""
    return re.findall(token_pattern, text)


def remove_stopwords(tokens: Iterable[str]) -> List[str]:
    """Remove English stopwords using scikit-learn's built-in list."""
    return [tok for tok in tokens if tok not in ENGLISH_STOP_WORDS]


# --------------------------------------------------------------------------- #
# Metadata stripping (for documents that look like Usenet posts)
# --------------------------------------------------------------------------- #
def strip_metadata(text: str) -> str:
    """Heuristically remove headers, quoted replies and signatures.

    This mirrors what ``fetch_20newsgroups(remove=('headers','footers','quotes'))``
    does for the training data, so pasted newsgroup posts are treated consistently.
    Ordinary documents (essays, reports) pass through almost unchanged.
    """
    if not isinstance(text, str):
        return ""
    lines = text.splitlines()

    # 1. Header block: lines before the first blank line, if most look like "Key: value".
    try:
        blank_idx = next(i for i, ln in enumerate(lines) if not ln.strip())
    except StopIteration:
        blank_idx = None
    if blank_idx is not None and blank_idx >= 2:
        head = lines[:blank_idx]
        header_like = sum(1 for ln in head if _HEADER_LINE_RE.match(ln) or ln.startswith((" ", "\t")))
        if header_like / len(head) >= 0.6:
            lines = lines[blank_idx + 1:]

    # 2. Signature: everything after a line that is exactly "--" / "-- ".
    for i, ln in enumerate(lines):
        if ln.strip() == "--":
            lines = lines[:i]
            break

    # 3. Quoted replies and attribution lines.
    kept = [
        ln for ln in lines
        if not ln.lstrip().startswith((">", "|")) and not _ATTRIBUTION_RE.match(ln.strip())
    ]
    return "\n".join(kept)


class TextPreprocessor:
    """Callable that turns raw document text into cleaned text ready for TF-IDF."""

    def __init__(self, strip_meta: bool = True):
        self.strip_meta = strip_meta

    def __call__(self, text: Any) -> str:
        if not isinstance(text, str):
            return ""
        if self.strip_meta:
            cleaned = clean_text(strip_metadata(text))
            if cleaned:
                return cleaned
            # Stripping removed everything -> fall back to the unstripped text.
        return clean_text(text)


# --------------------------------------------------------------------------- #
# TF-IDF
# --------------------------------------------------------------------------- #
def build_vectorizer(tfidf_cfg: Dict[str, Any]) -> TfidfVectorizer:
    """Create an (unfitted) TF-IDF vectorizer from the ``tfidf`` config section."""
    return TfidfVectorizer(
        max_features=tfidf_cfg.get("max_features", 15000),
        ngram_range=tuple(tfidf_cfg.get("ngram_range", (1, 2))),
        min_df=tfidf_cfg.get("min_df", 2),
        max_df=tfidf_cfg.get("max_df", 0.95),
        sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
        stop_words=tfidf_cfg.get("stop_words", "english"),
        token_pattern=tfidf_cfg.get("token_pattern", DEFAULT_TOKEN_PATTERN),
        lowercase=True,
        dtype=np.float32,
    )


def fit_vectorizer(train_texts: Iterable[str], tfidf_cfg: Dict[str, Any]) -> TfidfVectorizer:
    """Fit the vectorizer on TRAINING texts only."""
    vec = build_vectorizer(tfidf_cfg)
    vec.fit(train_texts)
    return vec


def save_vectorizer(vec: TfidfVectorizer, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, path)


def load_vectorizer(path: Path | str) -> TfidfVectorizer:
    return joblib.load(path)
