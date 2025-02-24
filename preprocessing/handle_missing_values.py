import pandas as pd

# Define file paths
input_file = "C:/Users/Dell/Desktop/Dig/data/predictive_maintenance.csv"
output_file = "C:/Users/Dell/Desktop/Dig/data/cleaned_data.csv"

# Load dataset
try:
    df = pd.read_csv(input_file)
    print("File loaded successfully.")
    
    # Handle missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # Save cleaned data
    df.to_csv(output_file, index=False)
    print("Missing values handled. Cleaned data saved successfully.")

except FileNotFoundError:
    print(f"Error: File {input_file} not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")
