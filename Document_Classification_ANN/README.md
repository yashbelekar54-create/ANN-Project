# Intelligent Document Classification System

**Design and Development of an Artificial Neural Network-Based Intelligent Document Classification System**
TF-IDF feature extraction + Multilayer Perceptron (TensorFlow/Keras) trained on the official 20 Newsgroups dataset, with a Streamlit web application.

---

## ⚠️ Delivery status (read first)

| Component | Status |
|---|---|
| Source code, Streamlit app, tests, notebooks, documentation | **Complete** |
| 20 Newsgroups dataset | **Not bundled.** Downloaded by one command (`python -m src.data_loader`) through the official scikit-learn loader |
| Trained ANN (`models/document_classifier.keras`) | **NOT included - not yet trained.** The packaging environment had no TensorFlow, no package-install access and no access to the dataset host, so a real training run was impossible. No model file was fabricated. |
| Evaluation results (`results/`) | **None yet.** They are produced by `python -m src.evaluate` after training |

Until you run the training step below, the app opens but shows *"Model not trained yet"* and refuses to classify. No accuracy figure appears anywhere in this project because none has been measured.
Expected effort: the whole pipeline is CPU-friendly; the 15,000-feature MLP trains on 11,314 x 0.85 documents with 64-sample mini-batches, which should be a matter of minutes on a laptop (not measured here).

---

## 1. Introduction
Organisations hold large numbers of text documents that must be sorted by topic. This project builds an automatic document classifier: text is converted to TF-IDF vectors and classified by an Artificial Neural Network into one of 20 newsgroup categories.

## 2. Problem statement
Manual document categorisation is slow, inconsistent and does not scale. Rule/keyword systems break on vocabulary variation. The task is to learn the mapping from document text to category from labelled examples with a neural network, and deliver it as a usable application.

## 3. Objectives
1. Prepare the real 20 Newsgroups corpus with a leakage-aware, reproducible pipeline.
2. Implement and train a genuine MLP (TensorFlow/Keras) - no pre-trained model, LLM, API or classical ML classifier as predictor.
3. Evaluate on an untouched test split and report honestly.
4. Provide a Streamlit app (single / batch classification of TXT, PDF, DOCX, history, explorer, performance).
5. Document and test the system.

## 4. Dataset
* **Source:** `sklearn.datasets.fetch_20newsgroups` ("bydate" split) - https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset
* About 18,846 documents (11,314 train + 7,532 test), 20 categories; labels are read from the dataset's own `target_names` and the class count is verified programmatically.
* Original train/test separation preserved; 15% of the training portion is held out as a stratified validation set (seed 42). The test set is never used for fitting or tuning.
* `python -m src.data_loader` writes `data/processed/newsgroups_processed.csv` (`document_id, document_text, category, category_id, cleaned_text, dataset_split` plus helper columns) and `results/metrics/dataset_summary.json` (counts, missing / empty / duplicate analysis, cleaning decisions). Nothing is silently removed.

## 5. System architecture
```
 Text / TXT / PDF / DOCX
        |  document_extractor.py  (PyMuPDF, python-docx)
        v
   preprocessing.py : strip metadata -> lowercase -> remove URLs/e-mails/punctuation -> normalise whitespace
        v
   TF-IDF (unigrams+bigrams, <=15,000 features, sparse float32)   [fit on training data only]
        v
   MLP: Dense 256 ReLU -> Dropout 0.4 -> Dense 128 ReLU -> Dropout 0.3 -> Dense 20 Softmax
        v
   predicted category + confidence + top-k probabilities  -> Streamlit UI / history
```

## 6. ANN explanation
A fully connected feed-forward network. The two ReLU hidden layers learn non-linear combinations of TF-IDF term weights; dropout (0.40 / 0.30) reduces over-fitting; the softmax output gives a probability for each of the 20 classes. Training: Adam (lr 0.001), sparse categorical cross-entropy, batch size 64, up to 40 epochs, early stopping on validation loss (patience 5, best weights restored), `ReduceLROnPlateau` scheduler, seed 42.

**Memory:** TF-IDF stays a SciPy CSR matrix. A `SparseBatchSequence` densifies one mini-batch at a time (64 x 15,000 x 4 bytes ~ 3.8 MB), so an 8 GB laptop is sufficient and no GPU is needed. Works on Google Colab.

## 7. Preprocessing explanation
* Training text = scikit-learn `remove=('headers','footers','quotes')` output, then `clean_text()` (lowercase, remove URLs/e-mails, keep `[a-z0-9]`, collapse whitespace).
* Headers/quotes can leak the label (e.g. a `Newsgroups:` line). `python -m src.train --compare-metadata` trains with and without metadata and compares them on the **validation** split.
* TF-IDF: `max_features=15000, ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True, stop_words="english", dtype=float32` (configurable in `config/config.yaml`).
* Documents entered in the app get the same cleaning, plus heuristic header/quote/signature stripping, so pasted Usenet posts behave like training data.
* Stopwords come from scikit-learn's built-in list; **NLTK is not needed**.

## 8. Technology stack
Python, TensorFlow/Keras, scikit-learn, Pandas, NumPy, SciPy, Matplotlib, Seaborn, Streamlit, Joblib, PyMuPDF, python-docx, PyYAML, pytest.

## 9. Project structure
```
Document_Classification_ANN/
├── app/            Home.py, pages/ (1-6), components/ui.py
├── config/         config.yaml
├── data/           raw/ (scikit-learn cache), processed/, README.md
├── models/         (created by training) document_classifier.keras, tfidf_vectorizer.joblib,
│                   label_encoder.joblib, training_config.json, category_mapping.json, preprocessing_config.json
├── notebooks/      01 EDA, 02 preprocessing, 03 training, 04 evaluation
├── results/        figures/, metrics/, reports/   (created by the scripts)
├── src/            data_loader, preprocessing, ann_model, train, evaluate, predict,
│                   document_extractor, history, categories, config
├── tests/          preprocessing, extractor, ann model, model loading, predictions
├── .streamlit/config.toml   light theme
├── requirements.txt  run_app.bat  run_pipeline.bat  PROJECT_REPORT.md  README.md
```

## 10. Installation (Windows 11)
Use **Python 3.10, 3.11 or 3.12** (64-bit). Open PowerShell in the project folder:
```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process RemoteSigned`.

## 11. Dataset acquisition
```powershell
python -m src.data_loader
```
Needs internet once (~14 MB download, cached in `data/raw`). Add `--force` to rebuild the CSV.

## 12. Model training
```powershell
python -m src.train                    # trains, saves models/ artifacts
python -m src.train --compare-metadata # additionally: with vs without metadata (validation split)
python -m src.train --epochs 3         # quick smoke run
```
(or run `notebooks/03_ann_model_training.ipynb`, also usable on Colab).

## 13. Model evaluation
```powershell
python -m src.evaluate
```
Evaluates on the untouched test split and writes `results/metrics/evaluation_metrics.json|csv`, `results/reports/classification_report.csv|txt`, `results/reports/results_summary.md`, `results/figures/confusion_matrix*.png`, `training_curves.png`, `per_category_metrics.png`, and `most_confused_pairs.csv`.

## 14. Running the application
```powershell
run_app.bat
```
or `streamlit run app\Home.py`. Pages: Home, Single Classification (text/TXT/PDF/DOCX, top-3/5 probabilities), Batch Classification (+CSV export), Category Explorer, Model Performance, Prediction History (metadata only, clearable), About.

Run everything in one go with `run_pipeline.bat`. Run tests: `python -m pytest tests -v`.

## 15. Testing status
Executed in the packaging environment (Python 3.13, no TensorFlow/Streamlit/PyMuPDF): all Python files compile; **19 tests passed, 5 skipped**. Passed: text cleaning, metadata stripping, TF-IDF sparsity/dtype/bigrams/joblib round-trip, split stratification, data-quality report, TXT and DOCX extraction, invalid/empty/corrupt file handling. **Skipped (could not run there):** PDF extraction tests (need PyMuPDF), ANN architecture/sparse-batch tests, trained-model loading and prediction tests (need TensorFlow and a trained model). They will run on your machine after installing the requirements and training; skipped tests are reported as skipped, never as passed. The Streamlit app itself was **not launched** in the packaging environment.

## 16. Screenshots
Add after running the app into `docs/screenshots/`: `home.png`, `single_classification.png`, `batch_classification.png`, `model_performance.png`.

## 17. Limitations
* Only the 20 newsgroup topics (1990s Usenet English); other documents are still forced into one of them.
* TF-IDF ignores word order/context; short documents are unreliable.
* No OCR for scanned PDFs (the app says so explicitly).
* Test accuracy is a property of this corpus and may not transfer to other document types.

## 18. Future scope
Word embeddings/transformer comparison, hyper-parameter search on the validation set, class-weighting, calibration of confidence, OCR support, multilingual support, explanation of predictions (top contributing terms), REST API.
