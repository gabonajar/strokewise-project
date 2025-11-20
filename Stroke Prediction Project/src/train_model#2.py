import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# === Load dataset ===
DATA_PATH = "../dat/healthcare-dataset-stroke-data-cleaned.csv"
df = pd.read_csv(DATA_PATH)

# Split features and target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Train/test split BEFORE SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Apply SMOTE ===
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("SMOTE applied:")
print("Before:", y_train.value_counts().to_dict())
print("After:", y_train_sm.value_counts().to_dict())

# === Define models ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(class_weight="balanced", probability=True),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=10  # helps with imbalance
    )
}

results = []
models_dir = "../models"

# Create folder if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# === Train, evaluate, save each model ===
for name, model in models.items():
    print(f"\nTraining {name}...")

    start = time.time()
    model.fit(X_train_sm, y_train_sm)
    end = time.time()

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion matrix for {name}:\n{cm}\n")

    # Save the model
    model_path = os.path.join(models_dir, f"{name.replace(' ', '_')}.pkl")
    joblib.dump(model, model_path)

    # Model size
    size_kb = os.path.getsize(model_path) / 1024

    # Save results
    results.append([name, accuracy, recall, f1, round(end - start, 3), round(size_kb, 2)])

# === Display results cleanly ===
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Recall", "F1-score", "Training Time (s)", "Model Size (KB)"]
)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

print("\n=== Final Model Comparison Table ===")
print(results_df.to_string(index=False))
