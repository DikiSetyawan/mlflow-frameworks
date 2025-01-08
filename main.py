from src.train import *
from src.monitor import *
from src.retrain import *




# if __name__ == "__main__":
#     linear = flexible_training(
#         model_name = 'Linear_regression',
#         run_name = f"Linear_regression{get_current_time()}",
#         fit_intercept = True, 
#         n_jobs = 2
#     )
    

#     xgboost = training_xg(
#         model_name = 'Xgboost_regression',
#         n_estimators = 100,
#         max_depth = 3,
#         learning_rate = 0.1,
#         run_name = f"Xgboost_regression{get_current_time()}"
#     )

#     monitor_model_xgboost(xgboost['model_uri'])
#     monitor_model_generic(linear['model_uri'])


from fastapi import FastAPI, Query
from pydantic import BaseModel
import time

# Assume the training and monitoring functions are defined somewhere
# flexible_training, training_xg, monitor_model_xgboost, monitor_model_generic

app = FastAPI()

def get_current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")

# Define a model for retraining parameters
class RetrainParams(BaseModel):
    model_uri: str
    additional_params: dict

# Endpoint for Linear Regression training
@app.get("/train/linear")
def train_linear(
    model_name: str = Query(..., description="The model name, e.g., Linear_regression"),
    run_name: str = Query(..., description="The unique run name"),
    fit_intercept: bool = Query(True, description="Whether to fit the intercept"),
    n_jobs: int = Query(2, description="The number of jobs for parallelism")
):
    linear = flexible_training(
        model_name=model_name,
        run_name=run_name,
        fit_intercept=fit_intercept,
        n_jobs=n_jobs
    )
    monitor_model_generic(linear['model_uri'])
    return {"status": "success", "model_uri": linear['model_uri']}


# Endpoint for XGBoost training
@app.get("/train/xgboost")
def train_xgboost(
    model_name: str = Query(..., description="The model name, e.g., Xgboost_regression"),
    run_name: str = Query(..., description="The unique run name"),
    n_estimators: int = Query(100, description="The number of trees to train"),
    max_depth: int = Query(3, description="The maximum depth of the trees"),
    learning_rate: float = Query(0.1, description="The learning rate")
):
    xgboost = training_xg(
        model_name=model_name,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        run_name=run_name
    )
    monitor_model_xgboost(xgboost['model_uri'])
    return {"status": "success", "model_uri": xgboost['model_uri']}


# Endpoint for retraining the models
@app.post("/retrain")
def retrain_model(expname:str):
    x = retraining(expname)
    return {"messages" : x}


import uvicorn 
if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=8000,reload = True)
    

