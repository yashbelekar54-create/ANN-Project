import argparse, joblib, pandas as pd
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--text", required=True)
p.add_argument("--skills", default="")
p.add_argument("--education", default="Bachelor")
p.add_argument("--experience", type=float, default=0)
p.add_argument("--threshold", type=float, default=.5)
a=p.parse_args()
base=Path(__file__).resolve().parents[1]
prep=joblib.load(base/"models/feature_preprocessor.joblib")
model=joblib.load(base/"models/multilabel_mlp_demo.joblib")
mlb=joblib.load(base/"models/role_binarizer.joblib")
row=pd.DataFrame([{"resume_text":a.text,"skills":a.skills,"education":a.education,"experience_years":a.experience}])
scores=model.predict_proba(prep.transform(row))
import numpy as np
if isinstance(scores,list): scores=np.column_stack([x[:,1] if x.ndim==2 else x for x in scores])
rank=sorted(zip(mlb.classes_,scores.reshape(-1)),key=lambda z:z[1],reverse=True)
chosen=[x for x in rank if x[1]>=a.threshold] or rank[:3]
for role,score in chosen: print(f"{role}: {score:.3f}")
