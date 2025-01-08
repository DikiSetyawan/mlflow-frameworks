import mlflow
import pandas as pd 
from sklearn.metrics import mean_squared_error, r2_score
from src.train import *
from src.helper import *
from mlflow.tracking import MlflowClient

mappings = {
    'Building Type': {'Residential': 1, 'Commercial': 2, 'Industrial': 3},
    'Day of Week': {"Weekday": 1, "Weekend": 0}}


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.autolog()



#model_name = 'Linear_regression_model_awal'

def monitor_model(model_uri):
    mlflow.set_experiment("monitoringFlow")
    data = pd.read_csv('/home/dikidwidasa/mlflow/data/dummy_data.csv')
    for col, map_dict in mappings.items():
        data = mapping(data, col, map_dict)
    X, y = feature_selection(data)

    model = mlflow.sklearn.load_model(model_uri)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    # Use a dynamic run name, for example, include the timestamp or some identifier
    run_name = f"Model_Monitor_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

# Uncomment to run the monitoring
# monitor_model(model_name)
