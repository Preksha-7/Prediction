import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
file_path = "../data/encoded_data.csv"  # Adjust path if needed
df = pd.read_csv(file_path)

# Create output directory for EDA images
output_dir = "../eda_plots"
os.makedirs(output_dir, exist_ok=True)

# Select only numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns
df_numeric = df[numeric_cols].dropna()  # Drop NaN values

# 1. Histogram of each numerical feature
df_numeric.hist(figsize=(12, 8), bins=20, edgecolor='black')
plt.suptitle("Feature Distributions")
plt.savefig(f"{output_dir}/histograms.png")
plt.close()

# 2. Box plot to detect outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_numeric)
plt.xticks(rotation=90)
plt.title("Box Plot of Numerical Features")
plt.savefig(f"{output_dir}/boxplot.png")
plt.close()

# 3. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig(f"{output_dir}/correlation_heatmap.png")
plt.close()

# 4. Pairplot for relationships (limit to 5 features if dataset is large)
if df_numeric.shape[1] > 5:
    selected_features = df_numeric.iloc[:, :5]  # Pick first 5 numerical features
else:
    selected_features = df_numeric

sns.pairplot(selected_features)
plt.savefig(f"{output_dir}/pairplot.png")
plt.close()

# 5. Failure Distribution (if 'failure' exists)
if 'failure' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df['failure'])
    plt.title("Failure Distribution")
    plt.savefig(f"{output_dir}/failure_distribution.png")
    plt.close()

# 6. Failure Type Distribution (if 'failure_type' exists)
if 'failure_type' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=df['failure_type'], order=df['failure_type'].value_counts().index)
    plt.xticks(rotation=45)
    plt.title("Failure Type Distribution")
    plt.savefig(f"{output_dir}/failure_type_distribution.png")
    plt.close()

print("✅ All EDA plots generated successfully! Check the 'eda_plots' folder.")
