import mlflow
import pandas as pd 
from sklearn.metrics import mean_squared_error
from src.train import *
from src.helper import *
from mlflow.tracking import MlflowClient

mappings = {
    'Building Type': {'Residential': 1, 'Commercial': 2, 'Industrial': 3},
    'Day of Week': {"Weekday": 1, "Weekend": 0}}



mlflow.set_tracking_uri("http://127.0.0.1:5000")

def get_latest_model_uri(model_name):
    
    # Initialize the MLflow client
    client = MlflowClient()

    # Get all versions of the registered model
    model_versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])

    # Find the latest version (highest version number)
    latest_version = max(model_versions, key=lambda mv: int(mv.version))

    # Get the URI for the latest version
    model_uri = latest_version.source

    print(f"Latest Model URI: {model_uri}")
    return model_uri


model_name = 'Linear_regression_model_awal'
#letest_model_uri = get_latest_model_uri(model_name)
#print(letest_model_uri)


def monitor_model(model_name):
    data = pd.read_csv('/home/dikidwidasa/mlflow/data/dummy_data.csv')
    for col, map_dict in mappings.items():
        data = mapping(data, col, map_dict)
    X, y = feature_selection(data)

    model_uri = get_latest_model_uri(model_name)
    model = mlflow.sklearn.load_model(model_uri)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    with mlflow.start_run(run_name="Model_Monitor") as run:
        mlflow.log_metric("mse", mse)

# if __name__ == "__main__":
#     monitor_model(model_name)


    
    