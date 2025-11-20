import os
import pandas as pd
import time
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Optional: xgboost
try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    print("XGBoost not installed. Skipping XGBClassifier.")
    xgboost_available = False
import pandas as pd

# Show all columns and full width in terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)   # adjust to your terminal width
pd.set_option('display.expand_frame_repr', False)

# -------------------------
# 1️⃣ Load cleaned dataset
# -------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dat_dir = os.path.join(project_root, "dat")
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

data_file = os.path.join(dat_dir, "healthcare-dataset-stroke-data-cleaned.csv")
df = pd.read_csv(data_file)
print("Dataset loaded:", df.shape)

# -------------------------
# 2️⃣ Split features/target
# -------------------------
X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# 3️⃣ Define models
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True)
}

if xgboost_available:
    models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

# -------------------------
# 4️⃣ Train, evaluate, save models
# -------------------------
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix for {name}:\n{cm}")

    # Save model
    model_file = os.path.join(models_dir, f"{name.replace(' ','_')}.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    model_size_kb = os.path.getsize(model_file) / 1024

    # Store results
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Recall": round(rec, 4),
        "F1-score": round(f1, 4),
        "Training Time (s)": round(training_time, 3),
        "Model Size (KB)": round(model_size_kb, 2)
    })

# -------------------------
# 5️⃣ Show summary table
# -------------------------
summary_df = pd.DataFrame(results).sort_values(by="Recall", ascending=False)
print("\n=== Summary Table ===")
print(summary_df.reset_index(drop=True))
