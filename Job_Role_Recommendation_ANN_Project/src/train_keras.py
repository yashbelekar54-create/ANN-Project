"""TensorFlow/Keras multi-label MLP template.
Prepare a real, licensed CSV with resume_text, skills, education,
experience_years, and semicolon-separated role_labels before running.
This template intentionally leaves dataset-specific split/feature fitting explicit.
"""
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from scipy.sparse import hstack

df = pd.read_csv("data/your_licensed_resume_dataset.csv")
df["labels"] = df["role_labels"].fillna("").apply(lambda s: [x.strip() for x in s.split(";") if x.strip()])
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["labels"])
Xtr, Xte, ytr, yte = train_test_split(df, Y, test_size=0.2, random_state=42)
# Fit all preprocessing on training data only.
text_vec = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words="english")
skill_vec = TfidfVectorizer(token_pattern=r"[^;]+", binary=True)
Xt = hstack([text_vec.fit_transform(Xtr.resume_text.fillna("")),
             skill_vec.fit_transform(Xtr.skills.fillna(""))]).toarray()
Xv = hstack([text_vec.transform(Xte.resume_text.fillna("")),
             skill_vec.transform(Xte.skills.fillna(""))]).toarray()
# Extend with structured fields in a production adaptation (education/experience).
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(Xt.shape[1],)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.35),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(Y.shape[1], activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")])
model.fit(Xt, ytr, validation_split=0.15, epochs=30, batch_size=32,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
print("Test-set score matrix:", model.predict(Xv[:5]))
