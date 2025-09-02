# pipeline_model.py
# Auto-generated pipeline based on best model selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network._multilayer_perceptron import MLPClassifier

def create_pipeline():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier())
    ])
    return pipeline

if __name__ == "__main__":
    pipe = create_pipeline()
    print("Pipeline created:", pipe)
