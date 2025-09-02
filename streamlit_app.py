
import streamlit as st
import pandas as pd
import joblib

pipeline = joblib.load("parkinsons_final/models/pipeline_model.joblib")

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

st.title("🧠 Parkinson’s Prediction App")
uploaded_file = st.file_uploader("📂 Upload CSV/XLSX", type=["csv","xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    st.dataframe(data.head())
    if st.button("🔮 Predict"):
        results = predict_with_risk(pipeline, data)
        st.dataframe(results)
        csv = results.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Download CSV", data=csv, file_name="predictions.csv", mime="text/csv")
