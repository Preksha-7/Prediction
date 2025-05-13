import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class PredictiveMaintenanceModel:
    def __init__(self, model_path=None):
        """
        Initialize the Predictive Maintenance model
        
        Args:
            model_path (str, optional): Path to a saved model file
        """
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
            print(f"Model loaded from {model_path}")
        else:
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
    
    def train(self, X_train, y_train):
        """
        Train the Random Forest model
        
        Args:
            X_train (DataFrame): Training features
            y_train (Series): Training target (failure indicator)
        """
        print("Training Predictive Maintenance model...")
        self.model.fit(X_train, y_train.values.ravel())
        print("Training complete.")
        
        # Calculate and store feature importances
        self.feature_importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    def predict(self, X):
        """
        Predict equipment failures
        
        Args:
            X (DataFrame): Input features
            
        Returns:
            ndarray: Binary predictions (1 for failure, 0 for no failure)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict failure probabilities
        
        Args:
            X (DataFrame): Input features
            
        Returns:
            ndarray: Probability of failure for each sample
        """
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (DataFrame): Test features
            y_test (Series): Test target
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Failure', 'Failure'],
                    yticklabels=['No Failure', 'Failure'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save confusion matrix
        os.makedirs('app/static/images', exist_ok=True)
        plt.savefig('app/static/images/confusion_matrix.png')
        plt.close()
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=self.feature_importances[:10])
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()
        plt.savefig('app/static/images/feature_importance.png')
        plt.close()
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm.tolist(),
            'feature_importances': self.feature_importances.to_dict(orient='records')
        }
    
    def save_model(self, filename='predictive_maintenance_model.pkl'):
        """
        Save the model to disk
        
        Args:
            filename (str): Filename to save the model
        """
        os.makedirs('models/saved', exist_ok=True)
        path = os.path.join('models/saved', filename)
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)
        print(f"Model saved to {path}")
        return path

if __name__ == "__main__":
    # Example usage
    print("Loading data...")
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y1_train.csv')  # Assuming y1 is the failure indicator
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y1_test.csv')
    
    # Train model
    model = PredictiveMaintenanceModel()
    model.train(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Save model
    model.save_model()