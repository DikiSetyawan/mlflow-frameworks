from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mlflow.sklearn import log_model
import mlflow
import pandas as pd
import xgboost as xgb
import numpy as np
from src.EDA import *
from src.helper import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


mlflow.set_tracking_uri("http://localhost:5000")

#mlflow.autolog()


mappings = {
    'Building Type': {'Residential': 1, 'Commercial': 2, 'Industrial': 3},
    'Day of Week': {"Weekday": 1, "Weekend": 0}
}


def load_and_preprocess(path):
    df = load_data(path)
    for col, map_dict in mappings.items():
        df = mapping(df, col, map_dict)
    return df

def splitting_data(df):
    x, y = feature_selection(df)
    return x,y



def feature_selection_and_engineering(df, test_size=0.2, random_state=42):
    x, y = feature_selection(df)
    return custom_train_test_split(x, y, test_size=test_size, random_state=random_state)

# def train_and_log_model (df, test_data_path, model_name = "Linear_regression_model_awal"):
#     # Split data 
#     X_train, X_test, y_train, y_test = feature_selection_and_engineering(df, test_size=0.2, random_state=42)

#     run_name = f"Linear_regression_model_awal_{get_current_time()}"
#     #mlflow, start model 
#     with mlflow.start_run(run_name=run_name)as run:
#         # Create and train model
#         model = LinearRegression()
#         model.fit(X_train, y_train)

#         #model predictions 
#         y_pred = model.predict(X_test)

#         #log metric
#         mse = mean_squared_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)

#         mlflow.log_metric("mse", mse)
#         mlflow.log_metric("r2", r2)
        
#         #log artefact
#         mlflow.log_artifact(local_path=test_data_path)

#         #log model
#         mlflow.sklearn.log_model(model, model_name)
        
#         #save run id
#         run_id = run.info.run_id
    
#     model_uri = f"runs:/{run_id}/outputs/{model_name}"
#     version_timestampt = get_current_time()
#     registered_model_name = f"{model_name}{version_timestampt}"
#     registered_model_version = mlflow.register_model(model_uri, registered_model_name)

#     print(f"Model registered: {registered_model_name}, version: {registered_model_version.version}")
#     return registered_model_name, registered_model_version.version


def flexible_training(
        model_name = 'Linear_regression',
        run_name = f"Linear_regression{get_current_time()}",
        fit_intercept = True, 
        n_jobs = 2
) :
    mlflow.set_experiment("testingMlflow")
    # X_train,X_test, y_train, y_test = feature_selection_and_engineering(df, test_size=test_size, random_state=random_state)
    train_df = load_and_preprocess('/home/dikidwidasa/mlflow/data/train.csv')
    test_df = load_and_preprocess('/home/dikidwidasa/mlflow/data/test.csv')
    X_train, y_train = splitting_data(train_df)
    X_test, y_test = splitting_data(test_df)
    run_name = run_name

    with mlflow.start_run(run_name=run_name) as run : 
        model = LinearRegression(
            fit_intercept=fit_intercept,
            n_jobs = n_jobs,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mlflow.log_metric('mse', mse)
        mlflow.log_metric('r2', r2)

        mlflow.sklearn.log_model(model, model_name)
        run_id = run.info.run_id
    
    model_uri = f"runs:/{run_id}/{model_name}"
    registered_model_version = mlflow.register_model(model_uri, model_name)

    print(f"Model registered: {model_name}, version: {registered_model_version.version}")
    result = {
        'model_uri' : model_uri, 
        'registered_model_name' : model_name, 
        'registered_model_version' : registered_model_version,
        'run_id' : run_id
    } 
    return result


def training_xg(
        model_name = 'Xgboost_regression',
        n_estimators = 100,
        max_depth = 3,
        learning_rate = 0.1,
        run_name = f"Xgboost_regression{get_current_time()}",
):
    # Disable autologging
    mlflow.autolog(disable=True)

    mlflow.set_experiment("testingMlflow")
    train_df = load_and_preprocess('/home/dikidwidasa/mlflow/data/train.csv')
    test_df = load_and_preprocess('/home/dikidwidasa/mlflow/data/test.csv')
    X_train, y_train = splitting_data(train_df)
    X_test, y_test = splitting_data(test_df)
    
    with mlflow.start_run(run_name=run_name) as run:
        xgb_model = xgb.XGBRegressor(
            n_estimators = n_estimators,
            max_depth = max_depth,
            learning_rate = learning_rate
        )
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric('mse', mse)
        mlflow.log_metric('r2', r2)
        mlflow.xgboost.log_model(xgb_model, model_name)
        run_id = run.info.run_id

        model_uri = f"runs:/{run_id}/{model_name}"
        registered_model_version = mlflow.register_model(model_uri, model_name)

        print(f"Model registered: {model_name}, version: {registered_model_version.version}")
        result = {
            'model_uri' : model_uri, 
            'registered_model_name' : model_name, 
            'registered_model_version' : registered_model_version,
            'run_id' : run_id
        } 
    return result

