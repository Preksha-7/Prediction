import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Auto-detect file path
script_dir = os.path.dirname(os.path.abspath(__file__))
X_path = os.path.join(script_dir, "../data/X.csv")
y1_path = os.path.join(script_dir, "../data/y1.csv")
y2_path = os.path.join(script_dir, "../data/y2.csv")

# Load datasets
X = pd.read_csv(X_path)
y1 = pd.read_csv(y1_path)
y2 = pd.read_csv(y2_path)

# Perform train-test split
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.2, random_state=42)

# Save train-test splits
X_train.to_csv(os.path.join(script_dir, "../data/X_train.csv"), index=False)
X_test.to_csv(os.path.join(script_dir, "../data/X_test.csv"), index=False)
y1_train.to_csv(os.path.join(script_dir, "../data/y1_train.csv"), index=False)
y1_test.to_csv(os.path.join(script_dir, "../data/y1_test.csv"), index=False)
y2_train.to_csv(os.path.join(script_dir, "../data/y2_train.csv"), index=False)
y2_test.to_csv(os.path.join(script_dir, "../data/y2_test.csv"), index=False)

print("Train-test split completed. Saved all datasets.")
