import pandas as pd
from sklearn.model_selection import train_test_split

def perform_train_test_split(X, y1, y2):
    X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=42)
    _, _, y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=42)
    
    return X_train, X_test, y1_train, y1_test, y2_train, y2_test

if __name__ == "__main__":
    X = pd.read_csv("../data/X.csv")
    y1 = pd.read_csv("../data/y1.csv")
    y2 = pd.read_csv("../data/y2.csv")

    X_train, X_test, y1_train, y1_test, y2_train, y2_test = perform_train_test_split(X, y1, y2)

    X_train.to_csv("../data/X_train.csv", index=False)
    X_test.to_csv("../data/X_test.csv", index=False)
    y1_train.to_csv("../data/y1_train.csv", index=False)
    y1_test.to_csv("../data/y1_test.csv", index=False)
    y2_train.to_csv("../data/y2_train.csv", index=False)
    y2_test.to_csv("../data/y2_test.csv", index=False)

    print("Train-test split completed. Data saved in /data folder.")
