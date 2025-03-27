import os
import pandas as pd

# Auto-detect file path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../data/predictive_maintenance.csv")

# Load dataset
df = pd.read_csv(data_path)

# Select only numeric columns to fill missing values with mean
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.mean()))

# Save cleaned dataset
output_path = os.path.join(script_dir, "../data/cleaned_data.csv")
df.to_csv(output_path, index=False)

print(f"Missing values handled successfully. Saved to {output_path}")
