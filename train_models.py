"""
Training Script for ML Assignment 2
Dataset: Dry Beans (UCI ML Repository ID: 602)
Trains 5 classification models, evaluates them, saves models and test data.
"""

import os
import urllib.request
import zipfile
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
)

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── 1. Download & load Dry Beans dataset ─────────────────────────────────
def load_data():
    """Download Dry Beans dataset from UCI and return a DataFrame."""
    zip_url = (
        "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
    )
    zip_path = os.path.join(BASE_DIR, "dry_bean.zip")
    extract_dir = os.path.join(BASE_DIR, "dry_bean_raw")

    if not os.path.exists(zip_path):
        print("Downloading Dry Beans dataset from UCI …")
        urllib.request.urlretrieve(zip_url, zip_path)

    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    # find the Excel file
    xlsx_path = None
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.endswith(".xlsx"):
                xlsx_path = os.path.join(root, f)
                break

    df = pd.read_excel(xlsx_path)
    print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ── 2. Preprocess ────────────────────────────────────────────────────────
def preprocess(df):
    """Separate features / target, encode labels, split, scale."""
    target_col = "Class"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # save scaler and label encoder for the Streamlit app
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    # save feature names
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.joblib"))

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


# ── 3. Train models ──────────────────────────────────────────────────────
def get_models():
    """Return dict of model_name → (model, needs_scaling)."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42),
            True,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=10, random_state=42),
            False,
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(n_neighbors=7),
            True,
        ),
        "Naive Bayes (Gaussian)": (
            GaussianNB(),
            False,
        ),
        "Random Forest (Ensemble)": (
            RandomForestClassifier(n_estimators=200, max_depth=15,
                                   random_state=42, n_jobs=-1),
            False,
        ),
    }


def evaluate(model, X_test, y_test, n_classes):
    """Compute all six evaluation metrics."""
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)
        auc = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="weighted"
        )
    except Exception:
        auc = float("nan")

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1": f1_score(y_test, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    df = load_data()
    (X_train, X_test, X_train_s, X_test_s, y_train, y_test) = preprocess(df)
    n_classes = len(np.unique(y_train))

    models = get_models()
    results = {}

    for name, (model, needs_scaling) in models.items():
        print(f"\nTraining {name} …")
        X_tr = X_train_s if needs_scaling else X_train
        X_te = X_test_s if needs_scaling else X_test
        model.fit(X_tr, y_train)

        # save model
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, os.path.join(MODEL_DIR, f"{safe_name}.joblib"))

        metrics = evaluate(model, X_te, y_test, n_classes)
        results[name] = metrics
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # ── results table ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).T
    results_df = results_df[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(results_df.to_string(float_format="%.4f"))

    # save results for README / app
    results_df.to_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"))

    # ── save test data (CSV) ──────────────────────────────────────────────
    feature_names = list(X_train.columns)
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["Class"] = y_test  # encoded label
    test_df["Class_Name"] = [
        joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib")).inverse_transform([v])[0]
        for v in y_test
    ]
    test_path = os.path.join(BASE_DIR, "test_data.csv")
    test_df.to_csv(test_path, index=False)
    print(f"\nTest data saved to {test_path}  ({test_df.shape[0]} rows)")
    print("All models saved to model/ folder.")


if __name__ == "__main__":
    main()
