import pandas as pd
from tqdm import tqdm
import yaml
import joblib
import numpy as np
import pandas as pd

def load_raw_data(data_path):
    """
    Load csv files and join into one dataframe.

    Parameters:
    ----------
    path_data : str
        Raw dataset location.

    Returns:
    -------
    raw_dataset : pd.DataFrame
        Loaded and joined data.
    """
    # create dataframe
    raw_dataset = pd.DataFrame()

    for i in tqdm(os.listdir(data_path)) :
        raw_dataset = pd.concat([pd.read_csv(data_path + i),raw_dataset])

    return raw_dataset


# Function to load data.
def load_data(config):
    """
    Load every set of data.

    Parameters:
    ----------
    config : dict
        The loaded configuration file.

    Returns:
    -------
    data_train, data_valid, data_test : pd.DataFrame
        The loaded data.
    """

    # Load the train set.
    X_train = joblib.load(config["path_train_set"][0])
    y_train = joblib.load(config["path_train_set"][1])

    # Load the valid set.
    X_valid = joblib.load(config["path_valid_set"][0])
    y_valid = joblib.load(config["path_valid_set"][1])

    # Load the test set.
    X_test = joblib.load(config["path_test_set"][0])
    y_test = joblib.load(config["path_test_set"][1])

    # Concatenate the X and y of each set.
    data_train = pd.concat([X_train, y_train], axis=1)
    data_valid = pd.concat([X_valid, y_valid], axis=1)
    data_test = pd.concat([X_test, y_test], axis=1)

    # Validate the proportion.
    num_all_data = int(data_train.shape[0]) + int(data_valid.shape[0]) + int(data_test.shape[0])
    print(f"Data train proportion : {len(X_train) / num_all_data}")
    print(f"Data valid proportion : {len(X_valid) / num_all_data}")
    print(f"Data test proportion  : {len(X_test) / num_all_data}")

    return data_train, data_valid, data_test

    
    
    