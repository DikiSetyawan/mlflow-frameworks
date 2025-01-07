from src.train import *
from src.monitor import *

if __name__ == "__main__":
    test_data_path = "/home/dikidwidasa/mlflow/data/test_energy_data.csv"
    df = load_and_preprocess(test_data_path)
    #train_and_log_model(df, test_data_path)
    model_uri =flexible_training(df, test_data_path)
    print(model_uri)
    #model_uri = 'runs:/533e369df5164901a580e5ec0b375d1f/Linear_regression_model_awal'
    monitor_model(model_uri)
    #mlflow.sklearn.load_model(model_uri)

    #monitor_model("Linear_regression_model_awal")

