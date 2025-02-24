import pandas as pd
import os

# File paths
input_file = "C:/Users/Dell/Desktop/Dig/data/cleaned_data.csv"
output_file = "C:/Users/Dell/Desktop/Dig/data/encoded_data.csv"

try:
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"❌ Input file not found: {input_file}")

    # Load dataset
    df = pd.read_csv(input_file)
    print("✅ File loaded successfully. First 5 rows:")
    print(df.head())

    # Encode categorical columns (example using label encoding)
    categorical_cols = df.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        print(f"🔄 Encoding categorical columns: {list(categorical_cols)}")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    print("✅ Encoding complete. Saving file...")

    # Save encoded data
    df.to_csv(output_file, index=False, encoding="utf-8")

    # Verify saved file
    if os.path.exists(output_file):
        print(f"✅ Encoded data saved successfully: {output_file}")
    else:
        print("❌ File not created. Check permissions.")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"⚠️ An unexpected error occurred: {e}")
