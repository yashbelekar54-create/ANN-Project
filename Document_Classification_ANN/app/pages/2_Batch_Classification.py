"""Batch document classification."""
import pandas as pd
import streamlit as st

from components.ui import config, require_model, setup_page
from src import history
from src.config import resolve
from src.document_extractor import ExtractionError, extract_text

setup_page("Batch Document Classification", "Upload several TXT, PDF or DOCX files and classify them in one operation.")
clf = require_model()
cfg = config()

files = st.file_uploader("Upload documents", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if st.button("Classify all documents", type="primary"):
    if not files:
        st.warning("Please upload at least one document.")
        st.stop()

    rows, texts, names = [], [], []
    progress = st.progress(0.0, text="Extracting text...")
    for i, f in enumerate(files, start=1):
        try:
            res = extract_text(f.name, f.getvalue(), cfg["inference"]["max_upload_mb"])
            texts.append(res.text)
            names.append(f.name)
        except ExtractionError as exc:
            rows.append({"Document": f.name, "Status": f"Failed: {exc}", "Predicted category": "", "Confidence": None,
                         "2nd choice": "", "3rd choice": ""})
        progress.progress(i / len(files), text=f"Extracted {i}/{len(files)}")

    records = []
    if texts:
        progress.progress(1.0, text="Running the ANN...")
        for name, text, res in zip(names, texts, clf.predict_many(texts, top_k=3)):
            if "error" in res:
                rows.append({"Document": name, "Status": f"Failed: {res['error']}", "Predicted category": "",
                             "Confidence": None, "2nd choice": "", "3rd choice": ""})
                continue
            top = res["top_k"]
            rows.append({
                "Document": name, "Status": "OK" if not res["warning"] else "Low signal",
                "Predicted category": res["predicted_category"], "Confidence": round(res["confidence"], 4),
                "2nd choice": f"{top[1]['category']} ({top[1]['probability']:.2f})",
                "3rd choice": f"{top[2]['category']} ({top[2]['probability']:.2f})",
            })
            records.append({"source": "batch", "document": name,
                            "predicted_category": res["predicted_category"], "confidence": res["confidence"]})
    progress.empty()

    table = pd.DataFrame(rows)
    ok = int((table["Status"] == "OK").sum())
    failed = int(table["Status"].str.startswith("Failed").sum())
    if failed:
        st.warning(f"{ok} classified, {failed} could not be processed (see the Status column).")
    else:
        st.success(f"All {ok} documents were classified.")
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.download_button("Download results as CSV", table.to_csv(index=False).encode("utf-8"),
                       file_name="batch_classification_results.csv", mime="text/csv")
    if records:
        history.add_records(resolve(cfg, "history_file"), records)
