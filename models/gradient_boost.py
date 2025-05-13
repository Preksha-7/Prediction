import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class RULEstimationModel:
    def __init__(self, model_path=None):
        """
        Initialize the Remaining Useful Life (RUL) estimation model
        
        Args:
            model_path (str, optional): Path to a saved model file
        """
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
            print(f"RUL model loaded from {model_path}")
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
    
    def train(self, X_train, y_train):
        """
        Train the Gradient Boosting model for RUL estimation
        
        Args:
            X_train (DataFrame): Training features
            y_train (Series/DataFrame): Training target (tool wear)
        """
        print("Training RUL Estimation model...")
        # Convert DataFrame to Series if it's not already
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]
            
        self.model.fit(X_train, y_train)
        print("RUL training complete.")
        
        # Calculate and store feature importances
        self.feature_importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    def predict(self, X):
        """
        Predict the Remaining Useful Life
        
        Args:
            X (DataFrame): Input features
            
        Returns:
            ndarray: Predicted RUL values
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (DataFrame): Test features
            y_test (Series/DataFrame): Test target
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Convert DataFrame to Series if it's not already
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.iloc[:, 0]
            
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"RUL Model Evaluation:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual RUL')
        plt.ylabel('Predicted RUL')
        plt.title('Actual vs Predicted RUL')
        plt.tight_layout()
        
        # Save plot
        os.makedirs('app/static/images', exist_ok=True)
        plt.savefig('app/static/images/rul_prediction.png')
        plt.close()
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=self.feature_importances[:10])
        plt.title('Top 10 Feature Importances for RUL Estimation')
        plt.tight_layout()
        plt.savefig('app/static/images/rul_feature_importance.png')
        plt.close()
        
        # Create histogram of prediction errors
        errors = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of RUL Prediction Errors')
        plt.tight_layout()
        plt.savefig('app/static/images/rul_error_distribution.png')
        plt.close()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'feature_importances': self.feature_importances.to_dict(orient='records')
        }
    
    def save_model(self, filename='rul_estimation_model.pkl'):
        """
        Save the model to disk
        
        Args:
            filename (str): Filename to save the model
        """
        os.makedirs('models/saved', exist_ok=True)
        path = os.path.join('models/saved', filename)
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)
        print(f"RUL model saved to {path}")
        return path
    
    def calculate_rul(self, X, max_life=300):
        """
        Calculate the Remaining Useful Life based on model predictions
        
        Args:
            X (DataFrame): Input features
            max_life (int): Maximum component life in time units
            
        Returns:
            ndarray: Estimated RUL values
        """
        # Predict the component wear/degradation
        wear_predictions = self.predict(X)
        
        # Convert wear to RUL (remaining useful life)
        # Assuming higher wear values mean less RUL
        rul = max_life - wear_predictions
        
        # Ensure RUL is not negative
        rul = np.maximum(rul, 0)
        
        return rul

if __name__ == "__main__":
    # Example usage
    print("Loading data for RUL estimation...")
    X_train = pd.read_csv('data/X_train.csv')
    
    # For demonstration, we'll use the 'Tool wear [min]' column as our target
    # In a real scenario, you might calculate this based on historical data
    # Create a synthetic target if the column doesn't exist
    if 'tool wear [min]' in X_train.columns:
        y_train = X_train['tool wear [min]']
        X_train = X_train.drop('tool wear [min]', axis=1)
    else:
        # For demo purposes, create synthetic tool wear data
        print("Creating synthetic tool wear data for demonstration")
        y_train = pd.Series(np.random.randint(0, 250, size=X_train.shape[0]))
    
    X_test = pd.read_csv('data/X_test.csv')
    if 'tool wear [min]' in X_test.columns:
        y_test = X_test['tool wear [min]']
        X_test = X_test.drop('tool wear [min]', axis=1)
    else:
        y_test = pd.Series(np.random.randint(0, 250, size=X_test.shape[0]))
    
    # Train model
    model = RULEstimationModel()
    model.train(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Save model
    model.save_model()