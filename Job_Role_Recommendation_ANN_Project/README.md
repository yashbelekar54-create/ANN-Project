# Intelligent Job Role Recommendation Using ANN

## Project overview
This project demonstrates resume-to-job-role recommendation as a **multi-label classification** problem. One candidate may suit several roles at once, so the output layer has one independent score per role rather than a single mutually-exclusive class.

## Dataset notice — read before interpreting results
The included `data/resumes_demo.csv` is a **synthetic demonstration dataset** generated for this project. Its labels are rule-generated from the skill templates in `data/job_roles.csv`. It is included so the notebook runs immediately without downloads or credentials. It is NOT a real public resume corpus, and its metrics must NOT be described as real-world model performance.

For a real experiment, obtain a public resume corpus with valid multi-label occupation annotations, verify its license and redistribution conditions, inspect label definitions and duplicates, and replace the demo CSV. A suitable candidate to investigate is the Florex Resume Corpus / Multi-label Resume Dataset; verify current availability, provenance, terms, and annotation quality directly from its publisher before use. Do not upload private resumes without informed consent.

## Folder structure
- `data/resumes_demo.csv` — resume text, structured profile fields and semicolon-separated multi-label targets.
- `data/job_roles.csv` — role descriptions and illustrative required skills.
- `data/skills.csv` — skill vocabulary.
- `notebooks/Job_Role_Recommendation_ANN.ipynb` — executed notebook with outputs and additional candidate examples at the bottom.
- `models/` — fitted demo preprocessing, MLP and label mapping.
- `results/metrics/demo_metrics.json` — held-out demo metrics.
- `src/train_keras.py` — TensorFlow/Keras MLP training template for a prepared real dataset.
- `src/predict.py` — command-line prediction using the saved scikit-learn demo artifacts.
- `app/` — reserved for a future interface; this project intentionally focuses on notebook-based software.

## Requirements
Python 3.10–3.12 recommended. Install:
```bash
pip install -r requirements.txt
```
For TensorFlow, use a Python version/platform supported by the TensorFlow release you install. The executed notebook uses scikit-learn's `MLPClassifier` to remain lightweight and reproducible on CPU. The separate Keras script demonstrates the requested TensorFlow architecture.

## Run on Windows
1. Extract the ZIP.
2. Open Command Prompt in the extracted project folder.
3. Create an environment: `py -m venv .venv`
4. Activate: `.venv\\Scripts\\activate`
5. Install: `pip install -r requirements.txt`
6. Start Jupyter: `jupyter notebook`
7. Open `notebooks/Job_Role_Recommendation_ANN.ipynb` and use **Run All**.

The notebook's saved outputs are already present. Re-running it will retrain the model and overwrite demo artifacts.

## CSV schema
### resumes_demo.csv
- `resume_id`: synthetic record identifier.
- `resume_text`: resume summary text.
- `skills`: semicolon-separated skill list.
- `education`: Diploma, Bachelor, or Master.
- `experience_years`: numeric years of experience.
- `role_labels`: semicolon-separated target roles; multiple labels per row.
- `source`, `annotation_type`: provenance warning.

### job_roles.csv
- `role_id`, `role_name`, `role_description`, `required_skills`, `source`.

### skills.csv
- `skill_id`, `skill_name`, `skill_category`.

## Architecture and preprocessing
1. Resume text is cleaned by TF-IDF vectorization (unigrams and bigrams, English stop-word removal).
2. The skill list is encoded as a binary token feature block.
3. Experience is scaled; education is one-hot encoded.
4. The feature blocks are concatenated into one sparse feature matrix.
5. A multilayer perceptron (128 and 64 hidden units, ReLU, Adam) learns nonlinear patterns.
6. Each output node represents one job role. Independent sigmoid/logistic outputs allow several roles to be recommended simultaneously.
7. A configurable threshold converts scores to selected recommendations; a top-k fallback avoids returning no roles.

## How the ANN works
A resume is transformed into a numerical vector. The input layer receives TF-IDF text and structured features. Hidden dense layers learn weighted combinations of those features; ReLU adds non-linearity. During training, the network compares its output scores with a multi-hot target vector and updates weights using backpropagation and Adam to reduce binary cross-entropy-like loss. At inference, each output score is interpreted as a role relevance score. Since roles are not mutually exclusive, sigmoid outputs are used instead of softmax.

## Evaluation
The notebook reports micro/macro precision, recall and F1, weighted F1, Hamming loss, sample Jaccard and exact-match/subset accuracy. Choose thresholds on a validation set, not on the final test set. Report per-role precision/recall/F1 as well.

## Limitations, fairness and responsible use
- Synthetic demonstration metrics do not establish real-world quality.
- Resume wording, geography, education access and career gaps can encode unfair proxies.
- Do not use this prototype to reject, rank or make employment decisions about people.
- Do not infer protected traits.
- Show recommendations as educational suggestions; let candidates review and correct extracted skills.
- Skill-gap output is a transparent comparison against the role taxonomy, not an ANN explanation.
- This is educational career guidance, not a guarantee of employment.
