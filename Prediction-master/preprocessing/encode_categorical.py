import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Auto-detect file path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../data/predictive_maintenance.csv")

# Load dataset
df = pd.read_csv(data_path)

# Handle missing values
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)

# Check if 'productID' column exists before encoding
if "Product ID" in df.columns:
    encoder = LabelEncoder()
    df["Product ID"] = encoder.fit_transform(df["Product ID"])
else:
    print("Warning: 'Product ID' column not found in dataset.")

# Save cleaned dataset
output_path = os.path.join(script_dir, "../data/encoded_data.csv")
df.to_csv(output_path, index=False)
print(f"Missing values handled. Saved to {output_path}")
