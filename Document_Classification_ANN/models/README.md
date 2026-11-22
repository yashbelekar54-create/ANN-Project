# models/

**This folder is intentionally empty of model files: the ANN has NOT been trained yet.**

Run `python -m src.train` from the project root. It creates:

* `document_classifier.keras` - the trained ANN
* `tfidf_vectorizer.joblib` - TF-IDF vectorizer fitted on training data only
* `label_encoder.joblib`, `category_mapping.json`
* `preprocessing_config.json`
* `training_config.json` - contains `"status": "trained"`, best epoch, duration, TensorFlow version

The application only treats the model as ready when all files exist **and** `training_config.json` says `"status": "trained"`.
