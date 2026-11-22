"""Single document classification."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from components.ui import config, metric_card, require_model, setup_page
from src import history
from src.config import resolve
from src.document_extractor import ExtractionError, extract_text
from src.predict import EmptyDocumentError
from src.categories import describe

setup_page("Single Document Classification", "Paste text or upload a TXT, PDF or DOCX file and classify it with the ANN.")
clf = require_model()
cfg = config()
inf = cfg["inference"]

tab_text, tab_file = st.tabs(["Paste text", "Upload file"])
with tab_text:
    pasted = st.text_area("Document text", height=260, placeholder="Paste the document text here...")
with tab_file:
    upload = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"], accept_multiple_files=False)
top_k = st.radio("Show top predictions", [3, 5], index=1, horizontal=True)

if st.button("Classify document", type="primary"):
    text, doc_name = "", "pasted_text"
    try:
        if upload is not None:
            res = extract_text(upload.name, upload.getvalue(), inf["max_upload_mb"])
            text, doc_name = res.text, upload.name
            st.info(f"Extracted {res.n_words:,} words from **{upload.name}**" + (f" ({res.n_pages} pages)" if res.n_pages else "") + ".")
            with st.expander("Preview extracted text"):
                st.text(text[:2000] + ("..." if len(text) > 2000 else ""))
        else:
            text = pasted
        if not text.strip():
            st.warning("Please paste some text or upload a document first.")
            st.stop()

        result = clf.predict(text, top_k=top_k)
    except ExtractionError as exc:
        st.error(str(exc))
        st.stop()
    except EmptyDocumentError as exc:
        st.warning(str(exc))
        st.stop()
    except Exception as exc:  # unexpected failure: show it instead of crashing the app
        st.error(f"Prediction failed: {exc}")
        st.stop()

    conf = result["confidence"]
    st.markdown(
        f'<div class="result-card"><div>Predicted category</div><div class="cat">{result["predicted_category"]}</div>'
        f'<div>{describe(result["predicted_category"])}</div></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Confidence", f"{conf*100:.1f}%")
    with c2:
        metric_card("Document words", f"{len(text.split()):,}")
    with c3:
        metric_card("Words known to model", f"{result['known_terms']:,}", "distinct TF-IDF features matched")

    if result["warning"]:
        st.warning(result["warning"])
    elif conf < inf["low_confidence_threshold"]:
        st.warning("The model is uncertain about this document (confidence below "
                   f"{inf['low_confidence_threshold']*100:.0f}%). Check the runner-up categories below.")
    if len(text.split()) < inf["min_words_warning"]:
        st.warning(f"Short documents (< {inf['min_words_warning']} words) give less reliable predictions.")
    if conf >= inf["low_confidence_threshold"] and not result["warning"]:
        st.success("Classification completed.")

    top_df = pd.DataFrame(result["top_k"]).rename(columns={"category": "Category", "probability": "Probability"})
    left, right = st.columns([1, 1])
    with left:
        show = top_df.copy()
        show["Probability"] = (show["Probability"] * 100).round(2).astype(str) + " %"
        st.dataframe(show, hide_index=True, use_container_width=True)
    with right:
        fig, ax = plt.subplots(figsize=(6, 3 + 0.2 * top_k))
        d = top_df.iloc[::-1]
        ax.barh(d["Category"], d["Probability"], color="#3949ab")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Top predicted categories")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    history.add_records(resolve(cfg, "history_file"), [{
        "source": "single", "document": doc_name,
        "predicted_category": result["predicted_category"], "confidence": conf,
    }])
