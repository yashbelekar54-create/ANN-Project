"""Home / Dashboard page."""
import streamlit as st

from components.ui import config, metric_card, read_json, results_path, setup_page
from src.predict import artifact_status

setup_page(
    "Intelligent Document Classification System",
    "Classify text documents into 20 topics with an Artificial Neural Network (TF-IDF + Multilayer Perceptron).",
)
cfg = config()
status = artifact_status(cfg)
info = read_json("training_config_file")
metrics = None
mpath = results_path("metrics", "evaluation_metrics.json")
if mpath.exists():
    import json
    metrics = json.loads(mpath.read_text(encoding="utf-8"))
summary = read_json("dataset_summary_file")

if status["ready"]:
    st.success("A trained ANN model is loaded and ready to classify documents.")
else:
    st.warning(
        "No trained model was found. The application is installed correctly but the ANN has not been trained yet. "
        "Run `python -m src.train` (see README) and reload this page."
    )

st.subheader("Purpose and working overview")
st.write(
    "The system reads a document (pasted text, TXT, PDF or DOCX), cleans the text, converts it into a sparse TF-IDF "
    "vector of unigrams and bigrams, and passes the vector through a trained feed-forward neural network. The network "
    "outputs a probability for each of the 20 newsgroup categories; the highest probability is the predicted category."
)
st.markdown(
    "`Document` -> `Text extraction` -> `Cleaning` -> `TF-IDF (1-2 grams)` -> `Dense 256 + ReLU` -> `Dropout 0.4` -> "
    "`Dense 128 + ReLU` -> `Dropout 0.3` -> `Dense 20 + Softmax` -> `Category + confidence`"
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Categories", "20", "20 Newsgroups topics")
with c2:
    total = f"{summary['total_documents']:,}" if summary else "18,846*"
    metric_card("Documents", total, "train + validation + test" if summary else "*official size; dataset not yet prepared")
with c3:
    metric_card("TF-IDF features", f"{info['n_features']:,}" if info else f"{cfg['tfidf']['max_features']:,}",
                "from trained model" if info else "configured maximum")
with c4:
    metric_card("Test accuracy", f"{metrics['accuracy']*100:.2f}%" if metrics else "Not measured",
                "untouched test split" if metrics else "run python -m src.evaluate")

left, right = st.columns(2)
with left:
    st.subheader("Dataset")
    st.write(
        "Official **20 Newsgroups** (bydate) corpus loaded through `sklearn.datasets.fetch_20newsgroups`. The original "
        "train/test separation is preserved; 15% of the training portion is held out (stratified) for validation. "
        "Headers, footers and quoted replies are removed so the model learns from the content, not from metadata."
    )
with right:
    st.subheader("ANN architecture")
    st.table({
        "Layer": ["Input", "Dense + ReLU", "Dropout", "Dense + ReLU", "Dropout", "Dense + Softmax"],
        "Size / rate": [f"{cfg['tfidf']['max_features']:,} (max) TF-IDF features", cfg["model"]["hidden_units"][0],
                        cfg["model"]["dropout"][0], cfg["model"]["hidden_units"][1], cfg["model"]["dropout"][1], 20],
    })

if metrics:
    st.subheader("Model performance summary (test set)")
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("Macro F1", f"{metrics['macro_f1']:.4f}")
    with m2:
        metric_card("Weighted F1", f"{metrics['weighted_f1']:.4f}")
    with m3:
        metric_card("Macro precision / recall", f"{metrics['macro_precision']:.3f} / {metrics['macro_recall']:.3f}")

st.subheader("Navigation")
n1, n2, n3 = st.columns(3)
with n1:
    st.page_link("pages/1_Single_Classification.py", label="Single Document Classification")
    st.page_link("pages/2_Batch_Classification.py", label="Batch Document Classification")
with n2:
    st.page_link("pages/3_Category_Explorer.py", label="Category Explorer")
    st.page_link("pages/4_Model_Performance.py", label="Model Performance")
with n3:
    st.page_link("pages/5_Prediction_History.py", label="Prediction History")
    st.page_link("pages/6_About.py", label="About")
