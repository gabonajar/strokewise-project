# src/reclean_dataset.py
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

# ----------------------------
# Paths (project-root safe)
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "dat" / "healthcare-dataset-stroke-data.csv"  # original raw dataset
CLEAN_PATH = PROJECT_ROOT / "dat" / "healthcare-dataset-stroke-data-cleaned.csv"

# ----------------------------
# Columns to KEEP (your list)
# ----------------------------
KEEP_COLS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke"
]

# ----------------------------
# Load raw data (safe)
# ----------------------------
if not RAW_DATA.exists():
    raise FileNotFoundError(f"Raw data file not found at: {RAW_DATA}\nPlease ensure the raw CSV is at this location.")

df_raw = pd.read_csv(RAW_DATA)
print("Raw dataset loaded:", df_raw.shape)

# ----------------------------
# Validate available columns
# ----------------------------
missing = [c for c in KEEP_COLS if c not in df_raw.columns]
if missing:
    raise ValueError(f"The following required columns are missing from raw data: {missing}")

# ----------------------------
# Select only the important columns
# ----------------------------
df = df_raw[KEEP_COLS].copy()
print("Selected important columns. Shape:", df.shape)

# ----------------------------
# Handle missing values
# ----------------------------
# Fill BMI missing values with median (numeric)
if "bmi" in df.columns:
    median_bmi = df["bmi"].median()
    df["bmi"].fillna(median_bmi, inplace=True)
    print(f"Filled missing 'bmi' with median: {median_bmi}")

# ----------------------------
# Convert categorical -> numeric
# ----------------------------
# gender and smoking_status: use one-hot encoding (creates numeric 0/1 columns)
cat_cols = [c for c in ["gender", "smoking_status"] if c in df.columns]

df = pd.get_dummies(df, columns=cat_cols, prefix_sep="_", drop_first=False)

# ----------------------------
# Convert any boolean columns to 0/1 ints
# ----------------------------
bool_cols = df.select_dtypes(include="bool").columns.tolist()
if bool_cols:
    df[bool_cols] = df[bool_cols].astype(int)
    print("Converted boolean columns to int:", bool_cols)

# ----------------------------
# Replace spaces in column names with underscores
# ----------------------------
df.columns = [c.replace(" ", "_") for c in df.columns]

# ----------------------------
# Backup existing cleaned file (if present)
# ----------------------------
if CLEAN_PATH.exists():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = CLEAN_PATH.with_name(f"{CLEAN_PATH.stem}_backup_{ts}{CLEAN_PATH.suffix}")
    CLEAN_PATH.replace(backup_path)  # move current cleaned to backup
    print(f"Existing cleaned file backed up to: {backup_path}")

# ----------------------------
# Save new cleaned CSV
# ----------------------------
df.to_csv(CLEAN_PATH, index=False)
print("✅ New cleaned dataset saved to:", CLEAN_PATH)
print("Cleaned dataset shape:", df.shape)

# ----------------------------
# Show first 5 rows
# ----------------------------
pd.set_option("display.max_columns", None)
print("\n=== First 5 rows of the new cleaned dataset ===")
print(df.head())
