import os
import pandas as pd

# Auto-detect file path
script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, "..\data\X_train.csv")

# Load dataset
df = pd.read_csv(train_path)
print(df.head())