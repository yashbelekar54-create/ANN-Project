"""Shared UI helpers for the Streamlit application."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.predict import DocumentClassifier, artifact_status  # noqa: E402

APP_NAME = "Intelligent Document Classification System"

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
html, body, [class*="css"] {color: #263238;}
h1, h2, h3 {color: #263238; font-weight: 600;}
.hero {background: linear-gradient(135deg, #3949ab 0%, #5c6bc0 100%); color: #fff; padding: 1.6rem 2rem;
       border-radius: 10px; margin-bottom: 1.2rem;}
.hero h1 {color: #fff; margin: 0 0 .3rem 0; font-size: 1.9rem;}
.hero p {margin: 0; opacity: .92;}
.metric-card {background: #fff; border: 1px solid #e3e6ee; border-left: 4px solid #3949ab; border-radius: 8px;
              padding: 14px 18px; margin-bottom: .6rem;}
.metric-card .label {font-size: .75rem; color: #5f6b7a; text-transform: uppercase; letter-spacing: .05em;}
.metric-card .value {font-size: 1.65rem; font-weight: 600; color: #263238; line-height: 1.3;}
.metric-card .note {font-size: .78rem; color: #78849a;}
.result-card {background: #f4f6fd; border: 1px solid #c5cae9; border-radius: 10px; padding: 1.2rem 1.5rem; margin: .8rem 0;}
.result-card .cat {font-size: 1.7rem; font-weight: 700; color: #3949ab;}
</style>
"""


def setup_page(title: str, subtitle: str = "") -> None:
    st.set_page_config(page_title=f"{title} | {APP_NAME}", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(f"**{APP_NAME}**")
        status = artifact_status()
        if status["ready"]:
            st.success("Trained ANN model loaded")
        else:
            st.warning("Model not trained yet")
        st.caption("TF-IDF + MLP (TensorFlow/Keras) on 20 Newsgroups")
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div>'
        f'<div class="note">{note}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading trained ANN model...")
def _load_classifier() -> DocumentClassifier:
    return DocumentClassifier(load_config())


def require_model() -> DocumentClassifier:
    """Return the cached classifier, or stop the page with clear instructions."""
    status = artifact_status()
    if not status["ready"]:
        st.error("The trained model is not available, so documents cannot be classified yet.")
        st.markdown(
            "Train the ANN first (from the project root):\n\n"
            "```\npython -m src.data_loader\npython -m src.train\npython -m src.evaluate\n```\n"
            f"Missing artifacts: `{', '.join(status['missing']) or 'none'}` - trained flag: `{status['trained']}`"
        )
        st.stop()
    return _load_classifier()


def config() -> Dict[str, Any]:
    return load_config()


def read_json(key: str) -> Optional[dict]:
    path = resolve(load_config(), key)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return None
    return None


def results_path(*parts: str) -> Path:
    return ROOT / "results" / Path(*parts)
