from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

ROOT=Path(__file__).resolve().parents[1]
MODELS=ROOT/"models"
model=keras.models.load_model(MODELS/"job_role_ann.keras")
preprocessor=joblib.load(MODELS/"preprocessor.joblib")
roles=json.loads((MODELS/"role_names.json").read_text(encoding="utf-8"))

candidate={
    "resume_text_clean":"python sql excel power bi statistics pandas data visualization dashboard",
    "education_level":"Bachelor's",
    "domain_interest":"Technology",
    "years_experience":1
}
features=preprocessor.transform(pd.DataFrame([candidate]))
if hasattr(features,"toarray"):
    features=features.toarray()
scores=model.predict(np.asarray(features,dtype="float32"),verbose=0)[0]
ranked=sorted(zip(roles,scores),key=lambda item:item[1],reverse=True)
threshold=.35
selected=[(role,float(score)) for role,score in ranked if score>=threshold][:5]
if not selected:
    selected=ranked[:3]
print("Recommended roles:")
for role,score in selected:
    print(f"{role:35s} score={score:.3f}")
