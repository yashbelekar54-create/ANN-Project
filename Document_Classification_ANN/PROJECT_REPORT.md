# Design and Development of an Artificial Neural Network-Based Intelligent Document Classification System

*B.Tech (Third Year), Artificial Intelligence and Data Science - Academic Project Report*

> **Status note.** All numerical results in Section 17 are intentionally not filled in: the ANN had not been trained when this report was generated (no TensorFlow, package installation or dataset access in the build environment). After running `python -m src.train` and `python -m src.evaluate`, copy the measured values from `results/reports/results_summary.md` and `results/metrics/` into Section 17. No figure in this report was invented.

## 1. Abstract
This project designs, implements and packages a document classification system that assigns text documents to one of 20 topic categories. Documents are cleaned and converted to sparse TF-IDF vectors (unigrams and bigrams) and classified by a Multilayer Perceptron built with TensorFlow/Keras. The official 20 Newsgroups dataset is used with its original train/test separation; headers, footers and quoted replies are removed to avoid label leakage, and the effect of this choice is measured on a validation split. A Streamlit application supports single and batch classification of text, TXT, PDF and DOCX files. The pipeline is memory-efficient (sparse matrices, mini-batch densification) and runs on a CPU laptop with 8 GB RAM.

## 2. Introduction
Automatic text categorisation underlies e-mail routing, news organisation, content moderation and digital libraries. Neural networks can learn non-linear relations between term weights and categories, and TF-IDF provides a compact, interpretable numeric representation of text.

## 3. Problem statement
Given the text of a document, predict which of 20 predefined categories it belongs to, using an ANN trained from labelled data, and make the model usable through a web interface.

## 4. Objectives
1. Build a reproducible, leakage-aware data pipeline on the real 20 Newsgroups corpus.
2. Implement and train a genuine MLP classifier.
3. Evaluate on an untouched test set with accuracy, precision, recall and F1.
4. Deliver a Streamlit application with file upload and batch processing.
5. Provide automated tests and complete documentation.

## 5. Existing system and limitations
Manual sorting is slow and inconsistent. Keyword/rule systems need hand-maintained lists and fail on synonyms and new vocabulary. Many tutorial classifiers evaluate on data containing message headers, which can inflate accuracy because headers reveal the group; they also often densify the entire TF-IDF matrix, which exhausts memory on small machines.

## 6. Proposed system
TF-IDF + MLP trained on header-free text, with sparse mini-batch training, saved artifacts that the application loads (no retraining at start-up), explicit handling of empty/unsupported files and a clear "model not trained" state.

## 7. Literature / background overview
* **Bag-of-words and TF-IDF** weight a term by its frequency in a document and its rarity across the corpus (Sparck Jones, 1972; Salton and Buckley, 1988). Sublinear TF scaling (1 + log tf) dampens the influence of very frequent terms.
* **Multilayer perceptrons** trained by back-propagation (Rumelhart, Hinton and Williams, 1986) are universal function approximators and are a standard baseline for text classification on sparse features.
* **Adam** (Kingma and Ba, 2015) adapts learning rates per parameter; **dropout** (Srivastava et al., 2014) randomly disables units during training to reduce over-fitting.
* **20 Newsgroups** originates from Lang (1995) and is a standard benchmark for text classification; scikit-learn provides a loader with an option to remove headers, footers and quotes.

## 8. Dataset description
Official 20 Newsgroups "bydate" corpus via `fetch_20newsgroups` (about 18,846 posts: 11,314 training, 7,532 test; 20 categories). Categories: alt.atheism, comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x, misc.forsale, rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey, sci.crypt, sci.electronics, sci.med, sci.space, soc.religion.christian, talk.politics.guns, talk.politics.mideast, talk.politics.misc, talk.religion.misc. Class counts, missing/empty/duplicate analysis and imbalance ratio are generated into `results/metrics/dataset_summary.json` and the EDA notebook. Splits: original test kept intact; 15% of the training portion is a stratified validation set (seed 42).

## 9. System requirements
**Hardware:** any modern CPU, 8 GB RAM, no GPU. **Software:** Windows 11 (or Linux/macOS), Python 3.10-3.12, packages in `requirements.txt`. Google Colab is also supported.

## 10. System architecture
Document -> text extraction (PyMuPDF / python-docx / plain text) -> cleaning -> TF-IDF -> MLP -> category, confidence, top-k. Training code (`src/train.py`) and the application (`app/`) are separate; the app only loads saved artifacts (`models/`).

## 11. Methodology
1. Download and verify the dataset (20 classes). 2. Build the processed CSV and data-quality report (nothing removed). 3. Split train/validation/test. 4. Fit TF-IDF on training documents only. 5. Train the MLP with early stopping. 6. Compare metadata/no-metadata on validation. 7. Evaluate once on the test set. 8. Save artifacts and deploy in the app.

## 12. Text preprocessing
Lowercasing; removal of URLs, e-mail addresses and punctuation; whitespace normalisation; tokenisation with the pattern `\b[a-z0-9]{2,}\b`; English stopword removal (scikit-learn list). Letters and digits are kept because tokens such as `486` or `x11` are topical. Headers, footers and quotes are removed from training text. For documents entered in the application, a heuristic stripper removes header blocks, quoted lines, attribution lines and signatures, falling back to the unstripped text if nothing remains.

## 13. TF-IDF feature extraction
`TfidfVectorizer(max_features=15000, ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True, stop_words="english", dtype=float32)`, fitted on training documents only and saved with Joblib. The matrix stays sparse (CSR); only the current mini-batch is converted to a dense array.

## 14. ANN architecture
Input (TF-IDF features, at most 15,000) -> Dense(256, ReLU) -> Dropout(0.40) -> Dense(128, ReLU) -> Dropout(0.30) -> Dense(20, Softmax). Any architecture change made during experimentation must be recorded in Section 17 together with its validation result. The exact layer table is saved to `results/reports/model_summary.txt` at training time.

## 15. Training methodology
Adam (learning rate 0.001), sparse categorical cross-entropy, accuracy metric, batch size 64, maximum 40 epochs, early stopping on validation loss (patience 5, best weights restored), ReduceLROnPlateau (factor 0.5, patience 2), seed 42. Training history, best epoch and duration are saved in `models/training_config.json` and `results/metrics/training_history.json`. The test set is never used in training or tuning.

## 16. Evaluation metrics
Accuracy, macro and weighted precision / recall / F1, per-class classification report, confusion matrix (counts and row-normalised), test cross-entropy loss, training-vs-validation curves, most-confused category pairs. Macro averages treat all classes equally; weighted averages account for class sizes.

## 17. Results and discussion
**Not yet available.** Insert here, from the generated files:
* Training summary (epochs run, best epoch, duration) - `models/training_config.json`.
* Metadata comparison on validation data - `results/metrics/metadata_comparison.json`.
* Test metrics - `results/metrics/evaluation_metrics.json`, `results/reports/classification_report.csv`.
* Figures - `results/figures/training_curves.png`, `confusion_matrix*.png`, `per_category_metrics.png`.
* Discussion: strongest/weakest classes, expected confusion among related groups (e.g. `comp.*`, `talk.politics.*`, the religion groups) *if* the measured confusion matrix shows it, signs of over-fitting in the curves, and measured effect of removing metadata.

## 18. Application modules
Home/Dashboard; Single Classification (paste/TXT/PDF/DOCX, confidence, top-3/5); Batch Classification (multi-file, results table, CSV export, per-file error status); Category Explorer; Model Performance (reads only measured results); Prediction History (timestamp, filename, category, confidence - contents never stored; clearable); About. Supporting modules: `document_extractor.py` (graceful errors, clear message for scanned PDFs), `predict.py` (`DocumentClassifier`, empty-document handling, untrained-state detection).

## 19. Testing
Automated tests in `tests/` cover cleaning, metadata stripping, TF-IDF properties, dataset split/quality report, file extraction (TXT, DOCX, PDF) and error handling, ANN architecture and sparse batching, model loading in a fresh process, and prediction validity (shape, probabilities sum to 1, top-k ordering, batch vs single consistency, empty documents). Tests that need the trained model skip with an explicit reason when it does not exist. Record the pass/fail output of `python -m pytest tests -v` from your own run here. In the build environment, 19 tests passed and 5 were skipped (PyMuPDF/TensorFlow unavailable); the Streamlit app was not launched there.

## 20. Limitations
Closed set of 20 topics from 1990s Usenet; bag-of-words ignores order; short or off-topic documents give unreliable predictions; no OCR; English only; reported accuracy applies to this corpus.

## 21. Future scope
Transformer/embedding models for comparison, hyper-parameter search, class weighting, confidence calibration, explainability (top contributing terms), OCR, multilingual support, REST API deployment.

## 22. Conclusion
The project delivers a complete, reproducible TF-IDF + MLP document classification pipeline and application designed for modest hardware. Final conclusions about accuracy must be drawn from the measured results after training and evaluation (Section 17).

## 23. References
1. Lang, K. (1995). NewsWeeder: Learning to Filter Netnews. *Proceedings of the 12th International Conference on Machine Learning.*
2. Sparck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation*, 28(1).
3. Salton, G. and Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5).
4. Rumelhart, D. E., Hinton, G. E. and Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323.
5. Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR.*
6. Srivastava, N. et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15.
7. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12.
8. scikit-learn documentation: The 20 newsgroups text dataset - https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset
9. scikit-learn documentation: `fetch_20newsgroups` - https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
10. TensorFlow / Keras documentation - https://www.tensorflow.org/ , https://keras.io/
