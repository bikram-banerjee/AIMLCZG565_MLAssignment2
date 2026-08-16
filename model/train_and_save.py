import io
import json
import os
import zipfile
import requests
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import joblib

# ------------------------------------------------------------------
# 1. LOAD DATA (handles nested ZIPs from UCI)
# ------------------------------------------------------------------
print("Downloading Bank Marketing dataset from UCI...")
url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
r = requests.get(url)
outer_zip = zipfile.ZipFile(io.BytesIO(r.content))

# Try finding directly first
csv_candidates = [n for n in outer_zip.namelist() if n.endswith("bank-full.csv")]

if csv_candidates:
    csv_file = csv_candidates[0]
    print(f"Found CSV directly: {csv_file}")
    df = pd.read_csv(outer_zip.open(csv_file), sep=";")
else:
    # Search inside nested ZIP files
    print("CSV not found directly. Searching nested ZIPs...")
    inner_zip_names = [n for n in outer_zip.namelist() if n.endswith(".zip")]
    
    df = None
    for zname in inner_zip_names:
        try:
            inner_zip = zipfile.ZipFile(io.BytesIO(outer_zip.read(zname)))
            inner_csv = [n for n in inner_zip.namelist() if n.endswith("bank-full.csv")]
            if inner_csv:
                print(f"Found CSV inside nested archive: {zname}/{inner_csv[0]}")
                df = pd.read_csv(inner_zip.open(inner_csv[0]), sep=";")
                break
        except zipfile.BadZipFile:
            continue
    
    if df is None:
        # Debug info
        print("Files in outer ZIP:", outer_zip.namelist())
        raise FileNotFoundError("Could not find bank-full.csv inside the downloaded archive.")

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ------------------------------------------------------------------
# 2. BASIC PREPROCESSING
# ------------------------------------------------------------------
df["y"] = df["y"].map({"yes": 1, "no": 0})

X = df.drop("y", axis=1)
y = df["y"]

# ------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Save test CSV for Streamlit app
test_df = X_test.copy()
test_df["y"] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print(f"Saved test_data.csv ({len(test_df)} rows)")

# ------------------------------------------------------------------
# 4. PREPROCESSOR
# ------------------------------------------------------------------
cat_cols = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome"
]
num_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    sparse_threshold=0,  # dense output so GaussianNB works
)

# ------------------------------------------------------------------
# 5. MODELS
# ------------------------------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "DecisionTree": DecisionTreeClassifier(
        random_state=42, class_weight="balanced"
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "GaussianNB": GaussianNB(),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    ),
}

metrics = {}

# ------------------------------------------------------------------
# 6. TRAIN · EVALUATE · SAVE
# ------------------------------------------------------------------
os.makedirs("model", exist_ok=True)

for name, clf in models.items():
    print(f"Training {name}...")
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics[name] = {
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "AUC": round(float(roc_auc_score(y_test, y_proba)), 4),
        "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_test, y_pred)), 4),
    }

    joblib.dump(pipe, f"model/{name}.joblib", compress=3)
    print(f"  → Saved model/{name}.joblib")

# Save metrics JSON
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nFinal Metrics:")
print(pd.DataFrame(metrics).T)