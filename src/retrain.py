from src.train import *
#from train import *
import mlflow 

mlflow.set_tracking_uri("http://localhost:5000")

from mlflow.tracking import MlflowClient

# Initialize the MLflow Client
client = MlflowClient()


def get_best_model(exp_name): 
    experiments = client.get_experiment_by_name(exp_name)
    if not experiments:
        return {f"No experiment found with name: {exp_name}"}
    
    experiment_id = experiments.experiment_id
    all_runs = client.search_runs([experiment_id])

    best_model = None
    best_mse = float('inf')

    for run in all_runs:
        mse = run.data.metrics.get('mse', None)
        if mse is not None and mse < best_mse:
            best_mse = mse
            best_model = {
                "run_id": run.info.run_id,
                "model_name": run.data.tags.get('mlflow.runName', 'Unnamed'),
                "mse": mse,
                "artifact_uri": run.info.artifact_uri
            }

    return best_model


def get_latest_model(exp_name):
    experiments = client.get_experiment_by_name(exp_name)
    if not experiments:
        return {f"No experiment found with name: {exp_name}"}
    
    experiment_id = experiments.experiment_id
    all_runs = client.search_runs([experiment_id], order_by=['start_time DESC'])

    if not all_runs:
        return {f"No runs found for experiment: {exp_name}"}

    latest_run = all_runs[0]
    return {
        "run_id": latest_run.info.run_id,
        "model_name": latest_run.data.tags.get('mlflow.runName', 'Unnamed'),
        "mse": latest_run.data.metrics.get('mse', None),
        "artifact_uri": latest_run.info.artifact_uri
    }



def retraining(expname): 
    best_model = get_best_model(expname)
    print(best_model)
    #mlflow.register_model(best_model['artifact_uri'], best_model['model_name'])
    last_model = get_latest_model(expname)
    print(last_model)
    if best_model['mse'] < last_model['mse']:    
        return {f"model doesnt need to be retrained"}
    else:
        flexible_training()
        training_xg()
        return {f"model retrained"}



x = retraining("testingMlflow")
print(x)


