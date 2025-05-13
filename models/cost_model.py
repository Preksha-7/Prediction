import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

class CostMaintenanceModel:
    def __init__(self, model_path=None):
        """
        Initialize the Cost Maintenance Model
        
        Args:
            model_path (str, optional): Path to a saved model file
        """
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
            print(f"Cost maintenance model loaded from {model_path}")
        else:
            self.model = LinearRegression()
    
    def train(self, features, costs):
        """
        Train the Cost Maintenance Model
        
        Args:
            features (DataFrame): Features including RUL predictions, failure probabilities, etc.
            costs (Series): Maintenance and repair costs
        """
        print("Training Cost Maintenance model...")
        self.model.fit(features, costs)
        print("Cost model training complete.")
        
        # Calculate and store feature coefficients
        self.feature_coefficients = pd.DataFrame({
            'feature': features.columns,
            'coefficient': self.model.coef_
        }).sort_values('coefficient', ascending=False)
    
    def predict(self, features):
        """
        Predict maintenance costs
        
        Args:
            features (DataFrame): Input features
            
        Returns:
            ndarray: Predicted costs
        """
        return self.model.predict(features)
    
    def evaluate(self, features_test, costs_test):
        """
        Evaluate cost model performance
        
        Args:
            features_test (DataFrame): Test features
            costs_test (Series): Actual costs
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        costs_pred = self.predict(features_test)
        
        # Calculate metrics
        mse = np.mean((costs_test - costs_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(costs_test - costs_pred))
        r2 = self.model.score(features_test, costs_test)
        
        print(f"Cost Model Evaluation:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Plot actual vs predicted costs
        plt.figure(figsize=(10, 6))
        plt.scatter(costs_test, costs_pred, alpha=0.5)
        plt.plot([costs_test.min(), costs_test.max()], [costs_test.min(), costs_test.max()], 'r--')
        plt.xlabel('Actual Costs')
        plt.ylabel('Predicted Costs')
        plt.title('Actual vs Predicted Maintenance Costs')
        plt.tight_layout()
        
        # Save plot
        os.makedirs('app/static/images', exist_ok=True)
        plt.savefig('app/static/images/cost_prediction.png')
        plt.close()
        
        # Plot feature coefficients
        plt.figure(figsize=(10, 6))
        sns.barplot(x='coefficient', y='feature', data=self.feature_coefficients)
        plt.title('Feature Coefficients for Cost Prediction')
        plt.tight_layout()
        plt.savefig('app/static/images/cost_feature_coefficients.png')
        plt.close()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'feature_coefficients': self.feature_coefficients.to_dict(orient='records')
        }
    
    def save_model(self, filename='cost_maintenance_model.pkl'):
        """
        Save the model to disk
        
        Args:
            filename (str): Filename to save the model
        """
        os.makedirs('models/saved', exist_ok=True)
        path = os.path.join('models/saved', filename)
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)
        print(f"Cost model saved to {path}")
        return path
    
    def optimize_maintenance_schedule(self, equipment_data, failure_probs, rul_estimates, 
                                     repair_costs, downtime_penalty, preventive_cost_factor=0.6):
        """
        Optimize maintenance schedules based on cost-benefit analysis
        
        Args:
            equipment_data (DataFrame): Equipment operational data
            failure_probs (ndarray): Predicted failure probabilities
            rul_estimates (ndarray): Estimated remaining useful life
            repair_costs (float/ndarray): Cost of repair after failure
            downtime_penalty (float): Cost per unit of unplanned downtime
            preventive_cost_factor (float): Preventive maintenance cost as a fraction of repair cost
            
        Returns:
            DataFrame: Optimized maintenance schedule
        """
        # Create empty schedule
        schedule = pd.DataFrame({
            'equipment_id': equipment_data['Type'] if 'Type' in equipment_data.columns else np.arange(len(equipment_data)),
            'failure_probability': failure_probs,
            'rul_estimate': rul_estimates,
            'recommended_action': None,
            'expected_cost': None,
            'maintenance_date': None
        })
        
        # Calculate costs and determine optimal strategy for each equipment
        for i in range(len(schedule)):
            # Cost of reactive maintenance (after failure)
            if isinstance(repair_costs, np.ndarray) or isinstance(repair_costs, list):
                reactive_cost = repair_costs[i] + downtime_penalty
            else:
                reactive_cost = repair_costs + downtime_penalty
            
            # Cost of preventive maintenance
            preventive_cost = repair_costs * preventive_cost_factor if isinstance(repair_costs, (int, float)) else repair_costs[i] * preventive_cost_factor
            
            # Expected cost of each strategy
            expected_reactive_cost = failure_probs[i] * reactive_cost
            expected_preventive_cost = preventive_cost
            
            # Choose strategy with lower expected cost
            if expected_preventive_cost < expected_reactive_cost:
                schedule.loc[i, 'recommended_action'] = 'Preventive Maintenance'
                schedule.loc[i, 'expected_cost'] = expected_preventive_cost
                
                # Schedule maintenance before component reaches end of life
                # Use RUL to determine date (assuming current date + RUL days)
                days_until_maintenance = max(0, rul_estimates[i] - 5)  # 5 days safety margin
                import datetime
                today = datetime.datetime.now()
                maintenance_date = today + datetime.timedelta(days=days_until_maintenance)
                schedule.loc[i, 'maintenance_date'] = maintenance_date.strftime('%Y-%m-%d')
            else:
                schedule.loc[i, 'recommended_action'] = 'Monitor'
                schedule.loc[i, 'expected_cost'] = expected_reactive_cost
                schedule.loc[i, 'maintenance_date'] = None
        
        return schedule

if __name__ == "__main__":
    # Example usage
    try:
        print("Loading data for cost modeling...")
        
        # For demonstration purposes, create synthetic data
        # In real scenario, this would come from your actual datasets
        np.random.seed(42)
        n_samples = 100
        
        # Create synthetic features related to cost prediction
        features = pd.DataFrame({
            'failure_probability': np.random.uniform(0, 1, n_samples),
            'rul_estimate': np.random.uniform(10, 300, n_samples),
            'component_age': np.random.uniform(0, 500, n_samples),
            'num_previous_repairs': np.random.randint(0, 10, n_samples)
        })
        
        # Create synthetic costs: combination of features with some noise
        costs = 500 + 1000 * features['failure_probability'] - 2 * features['rul_estimate'] + 50 * np.random.randn(n_samples)
        costs = pd.Series(costs)
        
        # Split into train and test
        from sklearn.model_selection import train_test_split
        features_train, features_test, costs_train, costs_test = train_test_split(
            features, costs, test_size=0.2, random_state=42
        )
        
        # Train model
        model = CostMaintenanceModel()
        model.train(features_train, costs_train)
        
        # Evaluate model
        metrics = model.evaluate(features_test, costs_test)
        
        # Save model
        model.save_model()
        
        # Demonstrate schedule optimization
        equipment_data = pd.DataFrame({
            'Type': ['Pump', 'Motor', 'Valve', 'Compressor', 'Fan'],
            'Age': [120, 300, 50, 200, 150]
        })
        failure_probs = np.array([0.15, 0.8, 0.05, 0.6, 0.3])
        rul_estimates = np.array([200, 20, 300, 50, 150])
        repair_costs = np.array([5000, 8000, 2000, 10000, 3000])
        downtime_penalty = 2000
        
        schedule = model.optimize_maintenance_schedule(
            equipment_data, failure_probs, rul_estimates, repair_costs, downtime_penalty
        )
        
        print("\nOptimized Maintenance Schedule:")
        print(schedule)
        
    except Exception as e:
        print(f"Error in cost model script: {e}")