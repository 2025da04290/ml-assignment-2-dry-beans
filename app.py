"""
Streamlit App – ML Assignment 2
Dry Beans Classification – 5 Models
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# ── configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dry Beans Classifier",
    page_icon="🫘",
    layout="wide",
)

STUDENT_NAME = "K Vishnu"
STUDENT_ID = "2025DA04290"

# ── styling ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }

    .hero {
        background: linear-gradient(135deg, #7a3b1e 0%, #b5651d 55%, #d98e3f 100%);
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 22px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.35);
    }
    .hero h1 {
        color: #ffffff;
        font-size: 2.1rem;
        margin: 0 0 6px 0;
        font-weight: 800;
    }
    .hero p {
        color: #fbe8d3;
        font-size: 1.05rem;
        margin: 0;
    }

    .student-card {
        background: #1c2029;
        border: 1px solid #333944;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 14px;
    }
    .student-card .label {
        color: #9aa4b2;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 2px;
    }
    .student-card .value {
        color: #f4f4f4;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .section-badge {
        display: inline-block;
        background: #b5651d;
        color: white;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }

    div[data-testid="stMetric"] {
        background: #1c2029;
        border: 1px solid #333944;
        border-radius: 10px;
        padding: 12px 14px 6px 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes_gaussian.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}

# models that were trained on scaled data
SCALED_MODELS = {"Logistic Regression", "K-Nearest Neighbors"}

# ── load artifacts ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    return scaler, le, feature_names


@st.cache_resource
def load_model(model_name):
    path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
    return joblib.load(path)


@st.cache_data
def load_metrics_summary():
    path = os.path.join(MODEL_DIR, "metrics_summary.csv", )
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


# ── helper: compute metrics ──────────────────────────────────────────────
def compute_metrics(model, X, y, n_classes):
    y_pred = model.predict(X)
    try:
        y_proba = model.predict_proba(X)
        auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = float("nan")

    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, average="weighted"),
        "Recall": recall_score(y, y_pred, average="weighted"),
        "F1": f1_score(y, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y, y_pred),
    }, y_pred


# ── sidebar ──────────────────────────────────────────────────────────────
st.sidebar.markdown(
    f"""
    <div class="student-card">
        <div class="label">Student</div>
        <div class="value">👤 {STUDENT_NAME}</div>
        <div class="label">BITS ID</div>
        <div class="value">🎓 {STUDENT_ID}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.title("🫘 Dry Beans Classifier")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** Dry Beans (UCI)\n\n"
    "16 features · 7 classes · 13,611 instances"
)
model_choice = st.sidebar.selectbox(
    "Select Model", list(MODEL_FILES.keys())
)
st.sidebar.markdown("---")
st.sidebar.markdown("Upload test data CSV to evaluate.")
st.sidebar.markdown("---")
st.sidebar.caption(f"ML Assignment 2 · {STUDENT_NAME} · {STUDENT_ID}")

# ── main area ────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
        <h1>🫘 Machine Learning Assignment 2</h1>
        <p>Classification Models on the Dry Beans Dataset &nbsp;·&nbsp; {STUDENT_NAME} ({STUDENT_ID})</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── overview table ───────────────────────────────────────────────────────
metrics_summary = load_metrics_summary()
if metrics_summary is not None:
    st.markdown('<span class="section-badge">OVERVIEW</span>', unsafe_allow_html=True)
    st.markdown("#### 📊 Model Comparison (Training Evaluation)")
    fmt_df = metrics_summary.copy()
    for c in fmt_df.columns:
        fmt_df[c] = fmt_df[c].apply(lambda x: f"{x:.4f}")
    st.dataframe(fmt_df, use_container_width=True)

st.markdown("---")

# ── file upload ──────────────────────────────────────────────────────────
st.markdown('<span class="section-badge">EVALUATE</span>', unsafe_allow_html=True)
uploaded = st.file_uploader("📤 Upload test_data.csv", type=["csv"])

if uploaded is not None:
    try:
        test_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.success(f"Loaded {test_df.shape[0]} rows × {test_df.shape[1]} columns")

    scaler, le, feature_names = load_artifacts()

    # determine target column
    target_col = None
    for cand in ["Class", "Class_Name", "target", "label"]:
        if cand in test_df.columns:
            target_col = cand
            break

    if target_col is None:
        st.error("CSV must contain a 'Class' or 'Class_Name' column as the target.")
        st.stop()

    # prepare features and target
    feature_cols = [c for c in feature_names if c in test_df.columns]
    if len(feature_cols) != len(feature_names):
        st.warning(
            f"Expected {len(feature_names)} features, found {len(feature_cols)} "
            "in the uploaded CSV. Missing columns will cause errors."
        )

    X = test_df[feature_names].copy()
    
    # encode target
    if target_col == "Class_Name":
        y = le.transform(test_df[target_col])
    elif target_col == "Class":
        y = test_df[target_col].values
    else:
        # try to transform with label encoder
        y = le.transform(test_df[target_col])

    # load selected model
    model = load_model(model_choice)

    # scale if needed
    if model_choice in SCALED_MODELS:
        X_proc = scaler.transform(X)
    else:
        X_proc = X.values

    # compute metrics
    metrics, y_pred = compute_metrics(model, X_proc, y, len(le.classes_))

    # ── display metrics ───────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    metric_cards = [
        (col1, "Accuracy", metrics["Accuracy"]),
        (col2, "AUC", metrics["AUC"]),
        (col3, "Precision", metrics["Precision"]),
        (col4, "Recall", metrics["Recall"]),
        (col5, "F1 Score", metrics["F1"]),
        (col6, "MCC", metrics["MCC"]),
    ]

    for col, label, val in metric_cards:
        col.metric(label, f"{val:.4f}")

    st.markdown("---")

    # ── confusion matrix ──────────────────────────────────────────────────
    st.markdown('<span class="section-badge">RESULTS</span>', unsafe_allow_html=True)
    st.markdown(f"#### 🔲 Confusion Matrix — {model_choice}")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_choice}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # ── classification report ─────────────────────────────────────────────
    st.markdown(f"#### 📝 Classification Report — {model_choice}")
    report = classification_report(
        y, y_pred, target_names=le.classes_, output_dict=True
    )
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.style.format("{:.4f}", subset=[
        "precision", "recall", "f1-score"
    ]), use_container_width=True)

    st.markdown("---")

    # ── sample predictions ────────────────────────────────────────────────
    st.markdown("#### 🔍 Sample Predictions")
    pred_names = le.inverse_transform(y_pred)
    actual_names = le.inverse_transform(y)
    sample = pd.DataFrame({
        "Actual": actual_names[:20],
        "Predicted": pred_names[:20],
    })
    st.dataframe(sample, use_container_width=True)

else:
    st.info("👆 Upload a CSV file to see model evaluation results.")
    st.markdown(
        "The expected CSV format matches `test_data.csv` in the repository — "
        "it should contain the 16 numeric feature columns plus a `Class` "
        "(encoded) or `Class_Name` (string) target column."
    )

st.markdown("---")
st.caption(f"Machine Learning Assignment 2 · Dry Beans Classification · {STUDENT_NAME} ({STUDENT_ID})")
