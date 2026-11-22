"""About page."""
import streamlit as st

from components.ui import config, read_json, setup_page

setup_page("About", "Project details, technologies, dataset and limitations.")
cfg = config()
info = read_json("training_config_file")

st.subheader("Project")
st.write(
    "**Design and Development of an Artificial Neural Network-Based Intelligent Document Classification System** - "
    "an academic project (B.Tech, Artificial Intelligence and Data Science)."
)

st.subheader("Technologies")
st.markdown(
    "- **TensorFlow / Keras** - the ANN (Multilayer Perceptron)\n"
    "- **scikit-learn** - dataset loader, TF-IDF vectorizer, splitting, metrics\n"
    "- **Pandas / NumPy / SciPy** - data handling and sparse matrices\n"
    "- **Matplotlib / Seaborn** - visualisations\n"
    "- **Streamlit** - web interface\n"
    "- **PyMuPDF / python-docx** - PDF and DOCX text extraction"
)

st.subheader("Dataset")
st.write(
    "Official 20 Newsgroups corpus (about 18,846 posts, 20 categories) via `sklearn.datasets.fetch_20newsgroups`. "
    "Original train/test separation preserved; validation is a stratified split of the training portion."
)

st.subheader("ANN architecture")
m = cfg["model"]
st.write(
    f"TF-IDF input ({cfg['tfidf']['max_features']:,} features max, unigrams + bigrams) -> Dense {m['hidden_units'][0]} (ReLU) -> "
    f"Dropout {m['dropout'][0]} -> Dense {m['hidden_units'][1]} (ReLU) -> Dropout {m['dropout'][1]} -> Dense 20 (Softmax). "
    f"Adam (lr {m['learning_rate']}), sparse categorical cross-entropy, batch size {m['batch_size']}, early stopping on validation loss."
)
if info:
    st.caption(f"Model trained {info['trained_at']} - best epoch {info['best_epoch']} - TensorFlow {info['tensorflow_version']}.")

st.subheader("Limitations")
st.markdown(
    "- The model only knows the 20 newsgroup topics (1990s Usenet discussions); documents outside these topics "
    "are still forced into one of the 20 categories.\n"
    "- Bag-of-words TF-IDF ignores word order and context.\n"
    "- Short documents give unreliable predictions; low confidence should be treated as uncertainty.\n"
    "- Scanned PDFs are not supported (no OCR).\n"
    "- Language: English only."
)
