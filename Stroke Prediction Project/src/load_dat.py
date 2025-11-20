import os
import shutil
import kagglehub

# 1. Download dataset
print("⬇️ Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
print("Dataset downloaded to:", path)

# 2. Create ./dat directory in project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dat_dir = os.path.join(project_root, "dat")
os.makedirs(dat_dir, exist_ok=True)

print("Saving files into:", dat_dir)

# 3. Copy all dataset files into ./dat
for file in os.listdir(path):
    source = os.path.join(path, file)
    destination = os.path.join(dat_dir, file)
    shutil.copy(source, destination)
    print(f"✔ Copied: {file}")

print("\n✅ Dataset successfully saved into ./dat folder!")
