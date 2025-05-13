from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, session
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import traceback
from datetime import datetime

# Import model classes
from models.random_forest import PredictiveMaintenanceModel
from models.gradient_boost import RULEstimationModel
from models.cost_model import CostMaintenanceModel

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('app/static/images', exist_ok=True)

# Load models
def load_models():
    models = {
        'failure': None,
        'rul': None,
        'cost': None
    }
    models_loaded = True
    
    try:
        # Load failure prediction model
        failure_model_path = 'models/saved/predictive_maintenance_model.pkl'
        if os.path.exists(failure_model_path):
            models['failure'] = PredictiveMaintenanceModel(model_path=failure_model_path)
        else:
            models['failure'] = PredictiveMaintenanceModel()
            models_loaded = False
            
        # Load RUL estimation model
        rul_model_path = 'models/saved/rul_estimation_model.pkl'
        if os.path.exists(rul_model_path):
            models['rul'] = RULEstimationModel(model_path=rul_model_path)
        else:
            models['rul'] = RULEstimationModel()
            models_loaded = False
            
        # Load cost optimization model
        cost_model_path = 'models/saved/cost_maintenance_model.pkl'
        if os.path.exists(cost_model_path):
            models['cost'] = CostMaintenanceModel(model_path=cost_model_path)
        else:
            models['cost'] = CostMaintenanceModel()
            models_loaded = False
            
    except Exception as e:
        print(f"Error loading models: {e}")
        traceback.print_exc()
        models_loaded = False
        
    return models, models_loaded

models, models_loaded = load_models()

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_data(df):
    """Basic data preprocessing"""
    # Convert column names to lowercase and standardize names
    df.columns = df.columns.str.lower().str.strip()
    
    # Map common column names to expected format
    column_mapping = {
        'air temp': 'air temperature [k]',
        'air temperature': 'air temperature [k]',
        'process temp': 'process temperature [k]',
        'process temperature': 'process temperature [k]',
        'rotation': 'rotational speed [rpm]',
        'rotational speed': 'rotational speed [rpm]',
        'tool wear': 'tool wear [min]'
    }
    
    # Rename columns if they exist with different names
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    
    # Handle missing values - replace with mean
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # Encode categorical columns if any
    categorical_cols = df.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df

# Routes
@app.route('/')
def index():
    return render_template('index.html', models_loaded=models_loaded)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
            
        file = request.files['file']
        
        # If user doesn't select a file, browser also submits an empty part
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Save file
            filename = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Read and process data
                df = pd.read_csv(filepath)
                
                # Store in session
                session['uploaded_file'] = filepath
                session['has_data'] = True
                
                flash('File uploaded successfully!', 'success')
                return redirect(url_for('data_preview'))
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'danger')
                return redirect(request.url)
        else:
            flash('Only CSV files are allowed', 'danger')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/data_preview')
def data_preview():
    if 'uploaded_file' not in session:
        flash('Please upload a data file first', 'warning')
        return redirect(url_for('upload_file'))
        
    try:
        filepath = session['uploaded_file']
        df = pd.read_csv(filepath)
        
        # Basic data statistics
        stats = {
            'rows': df.shape[0],
            'columns': df.shape[1],
            'missing_values': df.isnull().sum().sum(),
            'column_names': df.columns.tolist()
        }
        
        # Get first 10 rows for preview
        preview = df.head(10).to_html(classes='table table-striped table-sm', index=False)
        
        # Create histogram for a numeric column if available
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        histograms = {}
        
        if numeric_cols:
            for col in numeric_cols[:3]:  # Create histograms for first 3 numeric columns
                plt.figure(figsize=(6, 4))
                plt.hist(df[col].dropna(), bins=20)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.tight_layout()
                
                # Save to bytesio object
                img = BytesIO()
                plt.savefig(img, format='png')
                plt.close()
                img.seek(0)
                
                # Encode to base64 for display
                histograms[col] = base64.b64encode(img.getvalue()).decode('utf-8')
        
        return render_template('data_preview.html', stats=stats, preview=preview, 
                              histograms=histograms, numeric_cols=numeric_cols)
                              
    except Exception as e:
        flash(f'Error reading file: {str(e)}', 'danger')
        return redirect(url_for('upload_file'))

@app.route('/failure_prediction', methods=['GET', 'POST'])
def failure_prediction():
    if 'uploaded_file' not in session:
        flash('Please upload a data file first', 'warning')
        return redirect(url_for('upload_file'))
        
    if request.method == 'POST':
        try:
            # Load and preprocess the data
            filepath = session['uploaded_file']
            df = pd.read_csv(filepath)
            df = preprocess_data(df)
            
            # Get relevant features for prediction
            # Adjust as needed based on your model requirements
            required_features = ['air temperature [k]', 'process temperature [k]', 
                                'rotational speed [rpm]', 'torque [nm]', 'tool wear [min]']
            
            # Check for missing required features
            missing_features = [feat for feat in required_features if feat not in df.columns]
            if missing_features:
                flash(f'Missing required features: {missing_features}', 'warning')
                return redirect(url_for('failure_prediction'))
            
            # Use only required features for prediction
            X = df[required_features]
            
            # Make predictions
            failure_probs = models['failure'].predict_proba(X)
            predictions = (failure_probs > 0.5).astype(int)  # Convert probabilities to binary predictions
            
            # Create results DataFrame
            results_df = df.copy()
            results_df['failure_probability'] = failure_probs
            results_df['failure_prediction'] = predictions
            
            # Save results in session
            results_html = results_df.head(20).to_html(classes='table table-striped table-sm', index=False)
            session['failure_results'] = results_html
            session['failure_data'] = results_df.to_dict('records')
            
            # Create a histogram of failure probabilities
            plt.figure(figsize=(10, 6))
            plt.hist(failure_probs, bins=20, alpha=0.75)
            plt.title('Distribution of Failure Probabilities')
            plt.xlabel('Probability of Failure')
            plt.ylabel('Count')
            plt.axvline(x=0.5, color='r', linestyle='--', label='Decision Threshold')
            plt.legend()
            plt.tight_layout()
            
            # Save to static folder
            hist_path = 'app/static/images/failure_prob_dist.png'
            plt.savefig(hist_path)
            plt.close()
            
            # Get high risk equipment (probability > 0.7)
            high_risk = results_df[results_df['failure_probability'] > 0.7]
            high_risk_count = len(high_risk)
            
            flash('Failure predictions completed successfully!', 'success')
            return render_template('failure_prediction.html', 
                                  results=results_html,
                                  high_risk_count=high_risk_count,
                                  has_results=True)
                                  
        except Exception as e:
            flash(f'Error during prediction: {str(e)}', 'danger')
            traceback.print_exc()
            return redirect(url_for('failure_prediction'))
    
    return render_template('failure_prediction.html', has_results=False)

@app.route('/rul_estimation', methods=['GET', 'POST'])
def rul_estimation():
    if 'uploaded_file' not in session:
        flash('Please upload a data file first', 'warning')
        return redirect(url_for('upload_file'))
        
    if request.method == 'POST':
        try:
            # Load and preprocess the data
            filepath = session['uploaded_file']
            df = pd.read_csv(filepath)
            df = preprocess_data(df)
            
            # Get relevant features for RUL estimation
            # Adjust as needed based on your model requirements
            required_features = ['air temperature [k]', 'process temperature [k]', 
                                'rotational speed [rpm]', 'torque [nm]', 'tool wear [min]']
            
            # Check for missing required features
            missing_features = [feat for feat in required_features if feat not in df.columns]
            if missing_features:
                flash(f'Missing required features: {missing_features}', 'warning')
                return redirect(url_for('rul_estimation'))
            
            # Use only required features for prediction
            X = df[required_features]
            
            # Make RUL predictions
            max_life = 300  # Example maximum life in days
            rul_estimates = models['rul'].calculate_rul(X, max_life=max_life)
            
            # Create results DataFrame
            results_df = df.copy()
            results_df['rul_estimate'] = rul_estimates.astype(int)
            
            # Get failure probabilities if available
            if 'failure_data' in session:
                failure_data = pd.DataFrame(session['failure_data'])
                if 'failure_probability' in failure_data.columns:
                    results_df['failure_probability'] = failure_data['failure_probability']
            
            # Save results in session
            results_html = results_df.head(20).to_html(classes='table table-striped table-sm', index=False)
            session['rul_results'] = results_html
            session['rul_data'] = results_df.to_dict('records')
            
            # Create a histogram of RUL estimates
            plt.figure(figsize=(10, 6))
            plt.hist(rul_estimates, bins=20, alpha=0.75)
            plt.title('Distribution of Remaining Useful Life Estimates')
            plt.xlabel('Estimated RUL (days)')
            plt.ylabel('Count')
            plt.tight_layout()
            
            # Save to static folder
            hist_path = 'app/static/images/rul_dist.png'
            plt.savefig(hist_path)
            plt.close()
            
            flash('RUL estimation completed successfully!', 'success')
            return render_template('rul_estimation.html', 
                                  results=results_html,
                                  has_results=True)
                                  
        except Exception as e:
            flash(f'Error during RUL estimation: {str(e)}', 'danger')
            traceback.print_exc()
            return redirect(url_for('rul_estimation'))
    
    return render_template('rul_estimation.html', has_results=False)

@app.route('/cost_optimization', methods=['GET', 'POST'])
def cost_optimization():
    if 'uploaded_file' not in session or 'rul_data' not in session:
        flash('Please upload data and run RUL estimation first', 'warning')
        return redirect(url_for('rul_estimation'))
        
    if request.method == 'POST':
        try:
            # Get parameters from form
            repair_cost = float(request.form.get('repair_cost', 5000))
            downtime_penalty = float(request.form.get('downtime_penalty', 2000))
            preventive_cost_factor = float(request.form.get('preventive_cost_factor', 0.6))
            
            # Load data from session
            rul_data = pd.DataFrame(session['rul_data'])
            
            # Check if we have the necessary data
            if 'failure_probability' not in rul_data.columns:
                flash('Failure probability data is missing. Please run failure prediction first.', 'warning')
                return redirect(url_for('failure_prediction'))
                
            if 'rul_estimate' not in rul_data.columns:
                flash('RUL estimate data is missing. Please run RUL estimation first.', 'warning')
                return redirect(url_for('rul_estimation'))
            
            # Prepare data for cost optimization
            equipment_data = rul_data.copy()
            failure_probs = equipment_data['failure_probability'].values
            rul_estimates = equipment_data['rul_estimate'].values
            
            # Generate maintenance schedule
            maintenance_schedule = models['cost'].optimize_maintenance_schedule(
                equipment_data, 
                failure_probs, 
                rul_estimates, 
                repair_cost, 
                downtime_penalty, 
                preventive_cost_factor
            )
            
            # Calculate total costs
            preventive_count = (maintenance_schedule['recommended_action'] == 'Preventive Maintenance').sum()
            monitor_count = (maintenance_schedule['recommended_action'] == 'Monitor').sum()
            
            total_preventive_cost = (maintenance_schedule['expected_cost'] * 
                                    (maintenance_schedule['recommended_action'] == 'Preventive Maintenance')).sum()
            total_monitoring_cost = (maintenance_schedule['expected_cost'] * 
                                    (maintenance_schedule['recommended_action'] == 'Monitor')).sum()
            total_expected_cost = total_preventive_cost + total_monitoring_cost
            
            # Create cost summary
            cost_summary = {
                'preventive_count': preventive_count,
                'monitor_count': monitor_count,
                'total_preventive_cost': total_preventive_cost,
                'total_monitoring_cost': total_monitoring_cost,
                'total_expected_cost': total_expected_cost
            }
            
            # Generate pie chart for action distribution
            plt.figure(figsize=(8, 6))
            plt.pie([preventive_count, monitor_count], 
                   labels=['Preventive Maintenance', 'Monitor'], 
                   autopct='%1.1f%%',
                   colors=['#5cb85c', '#5bc0de'],
                   startangle=90)
            plt.title('Recommended Maintenance Actions')
            plt.axis('equal')
            
            # Save pie chart
            pie_path = 'app/static/images/maintenance_actions_pie.png'
            plt.savefig(pie_path)
            plt.close()
            
            # Convert maintenance schedule to HTML for display
            schedule_html = maintenance_schedule.head(20).to_html(
                classes='table table-striped table-hover', 
                index=False,
                formatters={
                    'expected_cost': lambda x: f'${x:.2f}',
                    'failure_probability': lambda x: f'{x:.2%}'
                })
            
            flash('Cost optimization completed successfully!', 'success')
            return render_template('cost_optimization.html',
                                  schedule_html=schedule_html,
                                  cost_summary=cost_summary,
                                  has_results=True,
                                  repair_cost=repair_cost,
                                  downtime_penalty=downtime_penalty,
                                  preventive_cost_factor=preventive_cost_factor)
                                  
        except Exception as e:
            flash(f'Error during cost optimization: {str(e)}', 'danger')
            traceback.print_exc()
            return redirect(url_for('cost_optimization'))
    
    # Default parameters
    default_params = {
        'repair_cost': 5000,
        'downtime_penalty': 2000,
        'preventive_cost_factor': 0.6
    }
    
    return render_template('cost_optimization.html', has_results=False, **default_params)

if __name__ == '__main__':
    app.run(debug=True)