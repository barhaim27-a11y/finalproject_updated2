import streamlit as st
import pandas as pd
import numpy as np
import os, joblib, json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
)

# ============================
# Load Dataset (with fallback)
# ============================
def load_dataset():
    if os.path.exists("parkinsons_final/data/parkinsons.csv"):
        return pd.read_csv("parkinsons_final/data/parkinsons.csv")
    elif os.path.exists("data/parkinsons.csv"):
        return pd.read_csv("data/parkinsons.csv")
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
        df = pd.read_csv(url)
        if "name" in df.columns:
            df = df.drop(columns=["name"])
        return df

# ============================
# Load Pipeline (with fallback)
# ============================
def load_or_train_pipeline():
    if os.path.exists("models/pipeline_model.joblib"):
        return joblib.load("models/pipeline_model.joblib"), "✅ Loaded pipeline_model.joblib from models/"
    elif os.path.exists("parkinsons_final/models/pipeline_model.joblib"):
        return joblib.load("parkinsons_final/models/pipeline_model.joblib"), "✅ Loaded pipeline_model.joblib from parkinsons_final/models/"
    else:
        st.warning("⚠️ pipeline_model.joblib not found. Training fallback RF model...")
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        df = load_dataset()
        X, y = df.drop("status", axis=1), df["status"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/pipeline_model.joblib")
        return pipeline, "✅ Trained fallback RandomForest pipeline"

pipeline, load_msg = load_or_train_pipeline()
st.sidebar.success(load_msg)

# ============================
# Prediction with Risk Labels
# ============================
def predict_with_risk(model, samples):
    preds = model.predict(samples)
    probs = model.predict_proba(samples)[:, 1]
    mapping = {0: "Healthy", 1: "Parkinson’s"}
    def risk_label(p):
        if p < 0.33: return "🟢 Low"
        elif p < 0.66: return "🟡 Medium"
        else: return "🔴 High"
    return pd.DataFrame({
        "Prediction": [mapping[p] for p in preds],
        "Probability": probs.round(3),
        "Risk": [risk_label(p) for p in probs]
    }, index=samples.index)

# ============================
# Streamlit Tabs
# ============================
st.title("🧠 Parkinson’s Prediction App (v19)")

tabs = st.tabs([
    "🔍 EDA", 
    "🤖 Model Results", 
    "🧪 Playground", 
    "🩺 Prediction", 
    "📊 Explainability", 
    "📜 Training Log"
])

# --- EDA Tab ---
with tabs[0]:
    st.header("Exploratory Data Analysis")
    df = load_dataset()
    st.write("Dataset shape:", df.shape)
    st.dataframe(df.head())

    fig, ax = plt.subplots()
    sns.countplot(x="status", data=df, palette="Set2", ax=ax)
    st.pyplot(fig)

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

# --- Model Results Tab ---
with tabs[1]:
    st.header("Model Leaderboard")
    if os.path.exists("parkinsons_final/assets/leaderboard.json"):
        leaderboard = json.load(open("parkinsons_final/assets/leaderboard.json"))
        st.json(leaderboard)
    else:
        st.warning("Leaderboard file not found.")

    # אם קיימות תמונות של ROC/PR curves
    for curve in ["roc_curve.png", "pr_curve.png", "learning_curve.png"]:
        path = f"parkinsons_final/assets/{curve}"
        if os.path.exists(path):
            st.image(path, caption=curve)
        else:
            st.warning(f"{curve} not found.")

# --- Playground Tab ---
with tabs[2]:
    st.header("Playground: Compare Models")
    st.write("כאן אפשר לבחור מודל אחד או כמה מודלים ולהשוות תוצאות")
    st.info("👉 תוסיף כאן UI לבחירת פרמטרים (sliders / textboxes)")

# --- Prediction Tab ---
with tabs[3]:
    st.header("Make Predictions")
    uploaded_file = st.file_uploader("📂 Upload CSV/XLSX", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        st.subheader("Uploaded Data")
        st.dataframe(data.head())
        if st.button("🔮 Predict"):
            results = predict_with_risk(pipeline, data)
            st.subheader("Predictions")
            st.dataframe(results)
            csv = results.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ Download CSV", data=csv, file_name="predictions.csv", mime="text/csv")

# --- Explainability Tab ---
with tabs[4]:
    st.header("Explainability (SHAP)")
    shap_path = "parkinsons_final/assets/shap_summary.png"
    if os.path.exists(shap_path):
        st.image(shap_path)
    else:
        st.warning("No SHAP summary plot found.")

# --- Training Log Tab ---
with tabs[5]:
    st.header("Training Log")
    log_path = "parkinsons_final/assets/training_log.csv"
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        st.dataframe(log_df)
    else:
        st.warning("No training log found.")
