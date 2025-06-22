# import joblib
# import numpy as np
# import json
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from sklearn.model_selection import train_test_split

# # === Load your dataset ===
# # Replace with your actual path or method
# import pandas as pd
# df = pd.read_csv("Processed_LiverDataset.csv")  # Replace with actual path

# # Features and labels
# target = "Result"
# X = df.drop(target, axis=1)  # Replace 'target' with your target column
# y = df[target]

# # Train-test split (same split you used for training)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # === Load your models ===
# models = {
#     "ensemble": joblib.load("./models/ensemble_model.pkl"),
#     "random_forest": joblib.load("./models/rf_model.pkl"),
#     "xgboost": joblib.load("./models/xgb_model.pkl"),
#     "svm": joblib.load("./models/svm_model.pkl"),
#     "knn": joblib.load("./models/knn_model.pkl"),
#     "mlpnn": joblib.load("./models/mlp_model.pkl"),
#     # "lgboost":joblib.load("./models/lgb_model.pkl")
# }

# # === Evaluate and collect metrics ===
# metrics_dict = {}

# for name, model in models.items():
#     y_pred = model.predict(X_test)
    
#     # Check if model supports predict_proba
#     if hasattr(model, "predict_proba"):
#         y_proba = model.predict_proba(X_test)[:, 1]
#     else:
#         # fallback for models without predict_proba (e.g., SVM with no probability=True)
#         from sklearn.preprocessing import LabelEncoder
#         y_proba = model.decision_function(X_test)
#         y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())  # scale to 0-1

#     metrics_dict[name] = {
#         "Accuracy": round(accuracy_score(y_test, y_pred), 4),
#         "Precision": round(precision_score(y_test, y_pred), 4),
#         "Recall": round(recall_score(y_test, y_pred), 4),
#         "F1-score": round(f1_score(y_test, y_pred), 4),
#         "AUC": round(roc_auc_score(y_test, y_proba), 4)
#     }

# # === Save to JSON in static folder ===
# with open("./static/model_metrics.json", "w") as f:
#     json.dump(metrics_dict, f, indent=4)

# print("✅ Metrics saved to static/model_metrics.json")

import joblib
import numpy as np
import json
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.model_selection import train_test_split

# === Load your dataset ===
df = pd.read_csv("Processed_LiverDataset.csv")  # Replace with actual path

# Features and labels
target = "Result"
X = df.drop(target, axis=1)
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Define model paths ===
model_paths = {
    "ensemble": "models/ensemble_model.pkl",
    "random_forest": "models/rf_model.pkl",
    "xgboost": "models/xgb_model.pkl",
    "svm": "models/svm_model.pkl",
    "knn": "models/knn_model.pkl",
    "mlpnn": "models/mlp_model.pkl"
    "lgboost" : "models/lgb_model.pkl"
}

# === Create folders if not exist ===
os.makedirs('static/roc_curves', exist_ok=True)

# === Evaluate models ===
metrics_dict = {}

for name, path in model_paths.items():
    model = joblib.load(path)
    
    # Predictions
    y_pred = model.predict(X_test)

    # Get probability scores
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())  # scale to 0-1

    # Calculate metrics
    metrics_dict[name] = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1-score": round(f1_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_score), 4)
    }

    # === Plot and save ROC curve ===
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'{name.upper()} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUC-ROC Curve for {name.upper()}')
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(f'static/roc_curves/{name}.png')
    plt.close()

# === Save metrics to JSON ===
with open("static/model_metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=4)

print("✅ Metrics saved to static/model_metrics.json")
print("✅ ROC curves saved in static/roc_curves/")
