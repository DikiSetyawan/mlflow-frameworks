from src.train import *

if __name__ == "__main__":
    test_data_path = "/home/dikidwidasa/mlflow/data/test_energy_data.csv"
    df = load_and_preprocess(test_data_path)
    train_and_log_model(df, test_data_path)

