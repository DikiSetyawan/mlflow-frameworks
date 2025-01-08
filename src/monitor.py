import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.tracking import MlflowClient
import mlflow
import pandas as pd 
from sklearn.metrics import mean_squared_error, r2_score
from src.train import *
from src.helper import *
from mlflow.tracking import MlflowClient

# Define mappings
mappings = {
    'Building Type': {'Residential': 1, 'Commercial': 2, 'Industrial': 3},
    'Day of Week': {"Weekday": 1, "Weekend": 0}
}

# Function to load and preprocess the data
def load_and_preprocess_data(file_path='/home/dikidwidasa/mlflow/data/valid.csv'):
    data = pd.read_csv(file_path)
    for col, map_dict in mappings.items():
        data = mapping(data, col, map_dict)
    X, y = feature_selection(data)
    return X, y

# Function to calculate metrics and log them to MLflow
def log_metrics(mse, r2, model_type):
    run_name = f"Model_Monitor_{model_type}_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

# Generic model monitoring function
def monitor_model_generic(model_uri, model_type="sklearn"):
    mlflow.set_experiment("monitoringFlow")

    # Load data
    X, y = load_and_preprocess_data()

    try:
        # Load model based on type
        if model_type == "xgboost":
            model = mlflow.xgboost.load_model(model_uri)
        else:
            model = mlflow.sklearn.load_model(model_uri)

        # Make predictions
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)

        # Log metrics
        log_metrics(mse, r2, model_type)
    except Exception as e:
        print(f"Error loading or predicting with model {model_type}: {e}")

# Specific model monitoring function for linear regression
def monitor_model_linear(model_uri):
    monitor_model_generic(model_uri, model_type="linear_regression")

# Specific model monitoring function for XGBoost
def monitor_model_xgboost(model_uri):
    monitor_model_generic(model_uri, model_type="xgboost")

# Example of calling the monitoring functions
# Uncomment to run the monitoring
# monitor_model_linear('model_uri_for_linear')
# monitor_model_xgboost('model_uri_for_xgboost')
