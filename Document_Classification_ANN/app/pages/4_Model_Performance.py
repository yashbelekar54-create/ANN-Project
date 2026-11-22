"""Model performance page - shows ONLY results measured by src.evaluate."""
import json

import pandas as pd
import streamlit as st

from components.ui import metric_card, read_json, results_path, setup_page

setup_page("Model Performance", "Metrics measured on the untouched 20 Newsgroups test split.")

mpath = results_path("metrics", "evaluation_metrics.json")
if not mpath.exists():
    st.warning(
        "No evaluation results exist yet. Train and evaluate the model first:\n\n"
        "```\npython -m src.train\npython -m src.evaluate\n```"
    )
    st.stop()

m = json.loads(mpath.read_text(encoding="utf-8"))
info = read_json("training_config_file")

cols = st.columns(4)
for col, (label, key) in zip(cols, [("Accuracy", "accuracy"), ("Macro F1", "macro_f1"),
                                    ("Weighted F1", "weighted_f1"), ("Test loss", "test_loss")]):
    with col:
        metric_card(label, f"{m[key]:.4f}")
cols = st.columns(4)
for col, (label, key) in zip(cols, [("Macro precision", "macro_precision"), ("Macro recall", "macro_recall"),
                                    ("Weighted precision", "weighted_precision"), ("Weighted recall", "weighted_recall")]):
    with col:
        metric_card(label, f"{m[key]:.4f}")
if info:
    st.caption(f"Trained for {info['epochs_run']} epochs (best epoch {info['best_epoch']}) in "
               f"{info['training_duration_seconds']} s with TensorFlow {info['tensorflow_version']}. "
               f"Test documents: {m['n_test_documents']:,}.")

tab1, tab2, tab3, tab4 = st.tabs(["Training curves", "Confusion matrix", "Per-category scores", "Metadata comparison"])
with tab1:
    p = results_path("figures", "training_curves.png")
    st.image(str(p)) if p.exists() else st.info("training_curves.png not found.")
with tab2:
    norm = st.toggle("Row-normalised", value=True)
    p = results_path("figures", "confusion_matrix_normalized.png" if norm else "confusion_matrix.png")
    st.image(str(p)) if p.exists() else st.info("Confusion matrix image not found.")
    pairs = results_path("metrics", "most_confused_pairs.csv")
    if pairs.exists():
        st.write("Most frequent confusions")
        st.dataframe(pd.read_csv(pairs), hide_index=True, use_container_width=True)
with tab3:
    rp = results_path("reports", "classification_report.csv")
    if rp.exists():
        rep = pd.read_csv(rp, index_col=0)
        cats = rep.drop(index=[i for i in ("accuracy", "macro avg", "weighted avg") if i in rep.index])
        st.dataframe(cats[["precision", "recall", "f1-score", "support"]].round(4), use_container_width=True)
        p = results_path("figures", "per_category_metrics.png")
        if p.exists():
            st.image(str(p))
with tab4:
    cp = results_path("metrics", "metadata_comparison.json")
    if cp.exists():
        comp = json.loads(cp.read_text(encoding="utf-8"))
        st.write(f"Scored on: {comp['scored_on']}")
        st.dataframe(pd.DataFrame(comp["results"]).T.round(4), use_container_width=True)
        p = results_path("figures", "metadata_comparison.png")
        if p.exists():
            st.image(str(p))
    else:
        st.info("Not run yet. Use `python -m src.train --compare-metadata`.")
