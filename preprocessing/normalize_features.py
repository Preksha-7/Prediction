import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_features(df):
    scaler = MinMaxScaler()
    numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                      'Torque [Nm]', 'Tool wear [min]']
    
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

if __name__ == "__main__":
    df = pd.read_csv("../data/encoded_data.csv")
    df = normalize_features(df)
    df.to_csv("../data/normalized_data.csv", index=False)
    print("Numerical features normalized and saved as normalized_data.csv.")
