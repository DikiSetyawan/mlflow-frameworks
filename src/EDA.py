import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(path):
    return pd.read_csv(path)

def split_dataset_pandas(df, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1, random_state=42):
  """
  Splits a pandas DataFrame into train, test, and validation sets.

  Args:
    df: The input pandas DataFrame.
    train_ratio: Proportion of data for the training set (default: 0.7).
    test_ratio: Proportion of data for the test set (default: 0.2).
    valid_ratio: Proportion of data for the validation set (default: 0.1).
    random_state: Seed for random number generator (default: 42).

  Returns:
    A tuple containing the training, test, and validation DataFrames.
  """

  if not 0 <= train_ratio <= 1 or not 0 <= test_ratio <= 1 or not 0 <= valid_ratio <= 1:
    raise ValueError("Ratios must be between 0 and 1.")

  if abs(train_ratio + test_ratio + valid_ratio - 1) > 1e-6:
    raise ValueError("Ratios must sum to 1.")

  # Shuffle the DataFrame
  df = df.sample(frac=1, random_state=random_state)

  total_size = len(df)
  train_size = int(train_ratio * total_size)
  test_size = int(test_ratio * total_size)

  train_df = df.iloc[:train_size].to_csv('/home/dikidwidasa/mlflow/data/train.csv',index = False)
  test_df = df.iloc[train_size:train_size + test_size].to_csv('/home/dikidwidasa/mlflow/data/test.csv', index = False)
  valid_df = df.iloc[train_size + test_size:].to_csv('/home/dikidwidasa/mlflow/data/valid.csv', index = False)

  return None

def mapping(df, colsname, map_var):
    df[colsname] = df[colsname].map(map_var)
    return df

def feature_selection(df): 
    x = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
    return x,y

def custom_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Splits data into training and testing sets.

    Args:
        X: Features (array-like).
        y: Target variable (array-like).
        test_size: Proportion of data to include in the testing set (default: 0.2).
        random_state: Controls the randomness of the split (default: None).

    Returns:
        X_train: Features for the training set.
        X_test: Features for the testing set.
        y_train: Target variable for the training set.
        y_test: Target variable for the testing set.
    """

    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

def sklearnSplit(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# df = load_data('/home/dikidwidasa/mlflow/data/test_energy_data.csv')

# mappings = {
#         'Building Type': {'Residential': 1, 'Commercial': 2, 'Industrial': 3},
#         'Day of Week': {"Weekday": 1, "Weekend": 0}
#     }

#     # Apply mappings
# for col, map_dict in mappings.items():
#     df = mapping(df, col, map_dict)

# x, y = feature_selection(df)

# X_train, X_test, y_train, y_test = sklearnSplit(x, y, test_size=0.2, random_state=42)
# data = {
#     "lenx": len(x),
#     "leny": len(y),
#     "X_trainlen": len(X_train),
#     "X_testlen": len(X_test),
#     "y_trainlen": len(y_train),
#     "y_testlen": len(y_test)

# }
# print(data)