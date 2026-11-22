# Intelligent Job Role Recommendation Using ANN

An educational multi-label classification project that analyzes resume text, skills, education, experience and domain interest to recommend multiple job roles.

## Included files

```text
Intelligent_Job_Role_Recommendation_ANN/
├── data/
│   ├── resume_profiles.csv
│   ├── resume_role_labels.csv
│   ├── occupation_catalog.csv
│   └── skill_taxonomy.csv
├── notebooks/
│   └── Intelligent_Job_Role_Recommendation_ANN.ipynb
├── src/
│   ├── train_model.py
│   └── predict.py
├── docs/
│   └── MODEL_EXPLANATION.md
├── models/                  # created after training
├── requirements.txt
└── README.md
```

## Important: dataset provenance and label quality

The included resume profiles are **synthetic teaching examples**, generated for this project. They are not real people's resumes and are not copied from a public resume website. `resume_role_labels.csv` contains explicit multi-label annotations generated with a transparent role-skill-overlap rule and a seeded primary role. The occupation catalog is an educational role/skill taxonomy, not an official occupational standard.

This choice makes the project immediately runnable without scraping personal data or relying on an unverified third-party dataset. It does **not** satisfy a claim that the model has been trained on publicly collected, human-validated resume-to-job annotations. For research-grade evaluation, replace the synthetic CSVs with legally usable, consented or appropriately licensed resume data and independently reviewed multi-label labels. Keep the same column structure.

## What the project does

- Cleans and vectorizes resume text using TF-IDF.
- Encodes education and domain interest.
- Scales years of experience.
- Builds a multi-hot target matrix (one column per job role).
- Trains a TensorFlow/Keras multilayer perceptron (MLP).
- Uses sigmoid outputs and binary cross-entropy for multi-label prediction.
- Reports micro/macro precision, recall and F1, Hamming loss, subset accuracy and per-role metrics.
- Ranks multiple roles and compares candidate skills against role skill requirements.

## Requirements

- Python 3.9–3.12 recommended
- Jupyter Notebook or Google Colab
- TensorFlow, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib

## Run locally

Open a terminal in this project directory:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook notebooks/Intelligent_Job_Role_Recommendation_ANN.ipynb
```

Run all cells from top to bottom. The notebook expects the `data/` folder to remain at the project root. If opened in Colab, upload the complete project folder or place the `data` folder beside the notebook and adjust `DATA_DIR` if necessary.

## Run training as a Python script

```bash
python src/train_model.py
```

This trains the MLP and creates `models/job_role_ann.keras`, `models/preprocessor.joblib`, and `models/role_names.json`.

## Run a sample prediction

After training:

```bash
python src/predict.py
```

Edit the sample profile inside `src/predict.py` to test another candidate.

## CSV data dictionary

### `resume_profiles.csv`
- `resume_id`: unique synthetic profile ID
- `candidate_name`: fictional sample name
- `education_level`: Diploma, Bachelor's, Master's or PhD
- `years_experience`: numeric experience value
- `domain_interest`: broad interest area
- `skills`: comma-separated skills
- `resume_text`: synthetic resume-style text

### `resume_role_labels.csv`
Long-form multi-label annotation table:
- `resume_id`
- `job_role`
- `is_relevant`: 1 means the role is a positive label, 0 means it is not

Pivot this table by `resume_id` and `job_role` to create the target matrix. A profile can have several positive roles.

### `occupation_catalog.csv`
- `job_role`
- `role_description`
- `core_skills`
- `skill_count`

### `skill_taxonomy.csv`
- `skill`
- `skill_group`

## Model architecture

Input features → Dense(256, ReLU) → Batch Normalization → Dropout(0.35) → Dense(128, ReLU) → Dropout(0.25) → Dense(64, ReLU) → Dense(number of roles, Sigmoid).

The output layer has one neuron per role. Scores are independently produced, so several roles can be selected. The baseline threshold is 0.50 in the evaluation section and 0.35 in the interactive recommendation helper. Thresholds should be tuned on validation data rather than selected to maximize test performance.

## Evaluation metrics

- **Micro precision/recall/F1:** aggregate all role decisions before computing the metric.
- **Macro precision/recall/F1:** compute each role metric independently and average roles equally.
- **Hamming loss:** fraction of individual label decisions that are wrong.
- **Subset accuracy:** fraction of profiles for which the complete predicted label set exactly matches the true set.
- **Per-role report:** shows precision, recall, F1 and support for each job role.

## Additional notebook snippets

At the end of the notebook, reusable examples demonstrate:
1. Changing candidate details.
2. Pasting resume text.
3. Comparing skill gaps for selected roles.
4. Changing the recommendation threshold.
5. Scoring multiple candidates in a batch.

## How the model works

The full beginner-friendly explanation is in [`docs/MODEL_EXPLANATION.md`](docs/MODEL_EXPLANATION.md) and in the final notebook section. In short, TF-IDF and encoded profile attributes turn the resume into a numeric vector. The MLP learns associations between these features and multi-hot job-role labels. Each sigmoid output gives a separate role score. The highest scoring roles are returned, and role skill lists are used for a simple gap analysis.

## Responsible-use statement

**Recommendations are educational career guidance and are not guaranteed employment outcomes.** This project must not be used as an automated hiring, rejection or candidate-ranking system. The included data and labels are synthetic and cannot establish real-world hiring validity.
