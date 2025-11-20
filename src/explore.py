import pandas as pd
import os

# 1️⃣ Path to dataset
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dat_dir = os.path.join(project_root, "dat")
data_file = os.path.join(dat_dir, "healthcare-dataset-stroke-data.csv")  # adjust if file name is different

# 2️⃣ Load dataset
df = pd.read_csv(data_file)

# 3️⃣ Quick exploration
print("\n=== First 5 rows ===")
print(df.head())

print("\n=== Dataset info ===")
print(df.info())

print("\n=== Missing values per column ===")
print(df.isnull().sum())

print("\n=== Basic statistics ===")
print(df.describe())
