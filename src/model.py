
# src/model.py

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.preprocessing import load_data, preprocess_dataframe
from src.feature_extraction import build_vectorizer, extract_features
from src.evaluation import evaluate_model
import pandas as pd

# === 1️⃣ Load & preprocess dataset ===
df = load_data("../data/dataset.csv")
df = preprocess_dataframe(df)

# === 2️⃣ Feature extraction ===
vectorizer = build_vectorizer(max_features=3000)
X = extract_features(vectorizer, df["clean_text"])
y = df["decision_label"]

# === 3️⃣ Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 4️⃣ Build & train model ===
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# === 5️⃣ Evaluate model ===
metrics = evaluate_model(model, X_test, y_test)
print(f"\nAccuracy: {metrics['accuracy']}\n")
print(pd.DataFrame(metrics["report"]).transpose())

# === 6️⃣ Save model & vectorizer properly ===
current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, "../results")
os.makedirs(results_dir, exist_ok=True)

model_path = os.path.join(results_dir, "logistic_model.pkl")
vectorizer_path = os.path.join(results_dir, "tfidf_vectorizer.pkl")

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"\n✅ Model saved at: {os.path.abspath(model_path)}")
print(f"✅ Vectorizer saved at: {os.path.abspath(vectorizer_path)}")
