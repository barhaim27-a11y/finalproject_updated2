import streamlit as st
import pandas as pd
import os, joblib

# ============================
# פונקציה לטעינת מודל עם fallback
# ============================
def load_or_train_pipeline():
    if os.path.exists("models/pipeline_model.joblib"):
        return joblib.load("models/pipeline_model.joblib"), "✅ Loaded pipeline_model.joblib from models/"
    elif os.path.exists("parkinsons_final/models/pipeline_model.joblib"):
        return joblib.load("parkinsons_final/models/pipeline_model.joblib"), "✅ Loaded pipeline_model.joblib from parkinsons_final/models/"
    else:
        st.warning("⚠️ pipeline_model.joblib not found. Training fallback RF model...")

        # אימון מהיר (fallback)
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
        df = pd.read_csv(url)
        if "name" in df.columns:
            df = df.drop(columns=["name"])

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

        return pipeline, "✅ Trained and saved fallback RandomForest pipeline"

# ============================
# טעינת המודל
# ============================
pipeline, load_msg = load_or_train_pipeline()
st.sidebar.success(load_msg)

# ============================
# כאן נשאר כל שאר הקוד של האפליקציה
# (EDA tab, Model Results tab, Playground tab, Prediction tab, Explainability tab, Training Log tab)
# ============================

# דוגמה לפונקציית prediction עם Risk Labels (נשארת כמו שהייתה אצלך)
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

# משם והלאה – ממשיך ה־UI המלא שלך בדיוק כמו בגרסה 16
