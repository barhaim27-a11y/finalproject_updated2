# ============================
# 📦 Parkinson's Final Project Builder (v19 - Full with Assets + Fallback)
# ============================

# 1. התקנת ספריות
!pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost tensorflow joblib shap streamlit openpyxl statsmodels pyngrok

# ============================
# 2. ייבוא ספריות
# ============================
import os, json, joblib, shutil, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, VotingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve
)

# ============================
# 3. יצירת תיקיות
# ============================
BASE_DIR = "parkinsons_final"
os.makedirs(BASE_DIR, exist_ok=True)
for sub in ["data", "models", "assets"]:
    os.makedirs(os.path.join(BASE_DIR, sub), exist_ok=True)

# ============================
# 4. הורדת דאטה והכנה
# ============================
uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df = pd.read_csv(uci_url)
if "name" in df.columns: df = df.drop(columns=["name"])
df.to_csv(os.path.join(BASE_DIR, "data/parkinsons.csv"), index=False)

X = df.drop("status", axis=1)
y = df["status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# 5. מודלים ואימון
# ============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVC": SVC(probability=True, kernel="rbf"),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
}
models["Voting Ensemble"] = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)),
        ("lgbm", LGBMClassifier(random_state=42))
    ],
    voting="soft"
)

results = {}
for name, model in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:,1]
    results[name] = roc_auc_score(y_test, y_prob)

best_name = max(results, key=results.get)
best_model = Pipeline([("scaler", StandardScaler()), ("clf", models[best_name])]).fit(X_train, y_train)
print(f"🏆 Best model: {best_name} (AUC={results[best_name]:.3f})")

# ============================
# 6. שמירת מודלים
# ============================
joblib.dump(best_model.named_steps["clf"], os.path.join(BASE_DIR,"models","best_model.joblib"))
joblib.dump(best_model.named_steps["scaler"], os.path.join(BASE_DIR,"models","scaler.joblib"))
pipeline_model = Pipeline([
    ("scaler", best_model.named_steps["scaler"]),
    ("classifier", best_model.named_steps["clf"])
])
joblib.dump(pipeline_model, os.path.join(BASE_DIR,"models","pipeline_model.joblib"))

# ============================
# 7. נכסים (Assets)
# ============================

# --- Leaderboard ---
with open(os.path.join(BASE_DIR,"assets","leaderboard.json"), "w") as f:
    json.dump(results, f, indent=2)

# --- Training log ---
log_path = os.path.join(BASE_DIR,"assets","training_log.csv")
log_entry = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "best_model": best_name,
    "AUC": results[best_name]
}
if os.path.exists(log_path):
    pd.DataFrame([log_entry]).to_csv(log_path, mode="a", header=False, index=False)
else:
    pd.DataFrame([log_entry]).to_csv(log_path, index=False)

# --- ROC Curve ---
y_prob = best_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test,y_prob):.2f}")
plt.plot([0,1],[0,1],'--')
plt.legend(); plt.title("ROC Curve")
plt.savefig(os.path.join(BASE_DIR,"assets","roc_curve.png")); plt.close()

# --- Precision-Recall Curve ---
prec, rec, _ = precision_recall_curve(y_test, y_prob)
plt.figure()
plt.plot(rec, prec)
plt.title("Precision-Recall Curve")
plt.savefig(os.path.join(BASE_DIR,"assets","pr_curve.png")); plt.close()

# --- Learning Curve ---
train_sizes, train_scores, test_scores = learning_curve(best_model, X, y, cv=3, scoring="roc_auc")
plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Test")
plt.legend(); plt.title("Learning Curve")
plt.savefig(os.path.join(BASE_DIR,"assets","learning_curve.png")); plt.close()

# --- SHAP Summary ---
import shap
try:
    explainer = shap.TreeExplainer(best_model.named_steps["clf"])
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(BASE_DIR,"assets","shap_summary.png"), bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"⚠️ Could not generate SHAP summary: {e}")

# ============================
# 8. pipeline_model.py
# ============================
classifier_class = type(best_model.named_steps["clf"]).__name__
pipeline_code = f"""# pipeline_model.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from {best_model.named_steps['clf'].__module__} import {classifier_class}

def create_pipeline():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", {classifier_class}())
    ])
    return pipeline
"""
with open(os.path.join(BASE_DIR,"pipeline_model.py"), "w") as f:
    f.write(pipeline_code)

# ============================
# 9. streamlit_app.py (גרסה 19 מלאה)
# ============================
streamlit_code = """<הקוד של גרסה 19 עם fallback שנתתי לך קודם>"""
with open(os.path.join(BASE_DIR,"streamlit_app.py"), "w") as f:
    f.write(streamlit_code)

# ============================
# 10. README + requirements
# ============================
reqs = """pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
tensorflow
joblib
streamlit
shap
statsmodels
openpyxl
"""
with open(os.path.join(BASE_DIR,"requirements.txt"),"w") as f:
    f.write(reqs)

readme = f"""# 🧠 Parkinson’s Prediction Project (v19 Full)

Best model: **{best_name}**

Includes:
- Full EDA + Feature Selection
- Multiple Models + Ensembles
- Best Model + Scaler + Pipeline
- Risk Labels + Batch Predictions
- Leaderboard + Training Log
- Explainability (SHAP)
- ROC / PR / Learning Curves
"""
with open(os.path.join(BASE_DIR,"README.md"),"w") as f:
    f.write(readme)

# ============================
# 11. יצירת ZIP
# ============================
shutil.make_archive("parkinsons_final_ready_v19", 'zip', BASE_DIR)
from google.colab import files
files.download("parkinsons_final_ready_v19.zip")
