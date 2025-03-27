import os
import pandas as pd

# Auto-detect file path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../data/normalized_data.csv")

# Load dataset
df = pd.read_csv(data_path)

# Correct column names
target_col = "Target"
failure_col = "Failure Type"

# Ensure the columns exist before dropping
if target_col in df.columns and failure_col in df.columns:
    X = df.drop(columns=[target_col, failure_col])
    y1 = df[target_col]
    y2 = df[failure_col]

    # Save feature and target datasets
    X.to_csv(os.path.join(script_dir, "../data/X.csv"), index=False)
    y1.to_csv(os.path.join(script_dir, "../data/y1.csv"), index=False)
    y2.to_csv(os.path.join(script_dir, "../data/y2.csv"), index=False)

    print("Feature and target variables split and saved.")
else:
    print(f"Columns '{target_col}' or '{failure_col}' not found in dataset.")
    print("Available columns:", df.columns)
