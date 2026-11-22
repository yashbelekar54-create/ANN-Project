from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

resumes = pd.read_csv(DATA / "resume_profiles.csv")
labels = pd.read_csv(DATA / "resume_role_labels.csv")
occupations = pd.read_csv(DATA / "occupation_catalog.csv")
roles = occupations["job_role"].tolist()
wide = labels.pivot(index="resume_id", columns="job_role", values="is_relevant").reindex(columns=roles).fillna(0)
df = resumes.set_index("resume_id").join(wide).reset_index()
df["resume_text_clean"] = df["resume_text"].fillna("").astype(str).str.lower()
df["education_level"] = df["education_level"].fillna("Unknown")
df["domain_interest"] = df["domain_interest"].fillna("Unknown")
df["years_experience"] = pd.to_numeric(df["years_experience"], errors="coerce")
X = df[["resume_text_clean","education_level","domain_interest","years_experience"]]
Y = df[roles].to_numpy(dtype="float32")
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=.2,random_state=SEED)
prep = ColumnTransformer([
    ("text", TfidfVectorizer(max_features=2500,ngram_range=(1,2),min_df=2),"resume_text_clean"),
    ("cat", OneHotEncoder(handle_unknown="ignore"),["education_level","domain_interest"]),
    ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),["years_experience"])
])
a=prep.fit_transform(X_train); b=prep.transform(X_test)
a=a.toarray() if hasattr(a,"toarray") else a
b=b.toarray() if hasattr(b,"toarray") else b
net=keras.Sequential([
    layers.Input(shape=(a.shape[1],)), layers.Dense(256,activation="relu"),
    layers.BatchNormalization(),layers.Dropout(.35),layers.Dense(128,activation="relu"),
    layers.Dropout(.25),layers.Dense(64,activation="relu"),
    layers.Dense(len(roles),activation="sigmoid")
])
net.compile(optimizer="adam",loss="binary_crossentropy",metrics=["binary_accuracy"])
net.fit(a.astype("float32"),y_train,validation_split=.2,epochs=40,batch_size=32,
        callbacks=[keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)],verbose=1)
pred=(net.predict(b.astype("float32"),verbose=0)>=.5).astype(int)
print("Micro F1:",f1_score(y_test,pred,average="micro",zero_division=0))
print(classification_report(y_test,pred,target_names=roles,zero_division=0))
net.save(MODELS/"job_role_ann.keras")
joblib.dump(prep,MODELS/"preprocessor.joblib")
(MODELS/"role_names.json").write_text(json.dumps(roles,indent=2),encoding="utf-8")
print("Artifacts saved in",MODELS)
