import pandas as pd

# Load dataset
file_path = "C:/Users/Dell/Desktop/Dig/data/encoded_data.csv"  
df = pd.read_csv(file_path)

# Print dataset info
print("✅ Dataset Loaded Successfully!")
print("First 5 rows:\n", df.head())
print("\nColumns in dataset:", df.columns.tolist())

# Column names to lowercase for consistency
df.columns = df.columns.str.lower()

# Checking if required columns exist
missing_cols = [col for col in ['failure', 'failure type'] if col not in df.columns]

if missing_cols:
    print(f"❌ Missing columns: {missing_cols}. Trying encoded columns instead.")
    
    # Adjust based on the actual columns available in the dataset
    try:
        y1 = df['failure type_no failure']  # Use encoded failure type column
        y2 = df['failure type_overstrain failure']  # Example, choose the relevant one
        X = df.drop(columns=['failure type_no failure', 'failure type_overstrain failure'])  # Drop target columns
    except KeyError as e:
        print(f"❌ Column not found: {e}")
else:
    # Split features and targets 
    X = df.drop(columns=['failure', 'failure type'])  # Features
    y1 = df['failure']  # Target 1
    y2 = df['failure type']  # Target 2

print("✅ Features and targets split successfully!")

# Ensure all variables are non-empty
if X.shape[0] > 0 and y1.shape[0] > 0 and y2.shape[0] > 0:
    print("Feature shape:", X.shape, "Target shapes:", y1.shape, y2.shape)
    
    # Save X, y1, and y2 to CSV files for further use
    X.to_csv("C:/Users/Dell/Desktop/Dig/data/X.csv", index=False)
    y1.to_csv("C:/Users/Dell/Desktop/Dig/data/y1.csv", index=False)
    y2.to_csv("C:/Users/Dell/Desktop/Dig/data/y2.csv", index=False)
    print("✅ X.csv, y1.csv, and y2.csv saved successfully!")
else:
    print("❌ One or more variables are empty. Please check the selection of features and targets.")
