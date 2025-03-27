import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Auto-detect file path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../data/encoded_data.csv")

# Load dataset
df = pd.read_csv(data_path)

# Correct numerical column names
numerical_cols = [
    "Air temperature [K]", 
    "Process temperature [K]", 
    "Rotational speed [rpm]", 
    "Torque [Nm]", 
    "Tool wear [min]"
]

# Ensure numerical columns exist in the dataset
existing_numerical_cols = [col for col in numerical_cols if col in df.columns]

if existing_numerical_cols:
    # Normalize numerical columns
    scaler = MinMaxScaler()
    df[existing_numerical_cols] = scaler.fit_transform(df[existing_numerical_cols])

    # Save normalized dataset
    output_path = os.path.join(script_dir, "../data/normalized_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Feature normalization completed. Saved to {output_path}")
else:
    print("No valid numerical columns found for normalization.")
