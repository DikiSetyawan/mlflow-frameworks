from src.train import *
from src.monitor import *




if __name__ == "__main__":
    linear = flexible_training(
        model_name = 'Linear_regression',
        run_name = f"Linear_regression{get_current_time()}",
        fit_intercept = True, 
        n_jobs = 2
    )
    

    xgboost = training_xg(
        model_name = 'Xgboost_regression',
        n_estimators = 100,
        max_depth = 3,
        learning_rate = 0.1,
        run_name = f"Xgboost_regression{get_current_time()}"
    )

    monitor_model_xgboost(xgboost['model_uri'])
    monitor_model_generic(linear['model_uri'])

    
