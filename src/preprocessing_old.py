import os

import yaml
import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler
)

from utils import load_config
from load import load_data



# Function to join categories.
def join_categories(set_data, config):
    # Check if label found in set data.
    if config["label"] in set_data.columns.to_list():
        set_data = set_data.copy()

        # Rename SEDANG to TIDAK SEHAT.
        set_data["category"] = set_data["category"].replace("SEDANG", "TIDAK SEHAT")

        # Rename TIDAK SEHAT to TIDAK BAIK.
        set_data["category"] = set_data["category"].replace("TIDAK SEHAT", "TIDAK BAIK")

        return set_data
    else:
        raise RuntimeError("Label is not detected in the dataset.")

def nan_replace(set_data):
    set_data = set_data.copy()
    set_data = set_data.replace(-1, np.nan)
    return set_data


def fit_ohe_encoder(X_stasiun):
    """
    Fit the OHE encoder
    
    Parameters:
    ----------
    X_stasiun : pd.DataFrame
        Categorical input data
    
    Returns:
    -------
    ohe_encoder : sklearn object
        Fitted OHE encoder object
    """
    
    ohe_encoder = OneHotEncoder(sparse_output=False)
    ohe_encoder.fit(np.array(X_stasiun).reshape(-1, 1))
    
    # Serialize the ohe_encoder object.    
    joblib.dump(ohe_encoder, '../models/ohe_stasiun.pkl')
    
    return ohe_encoder

def transform_ohe_encoder(set_data, transformed_column, ohe_path):
    """
    Transform the categorical input data using OHE encoder
    
    Parameters:
    ----------
    set_data : pd.DataFrame
        Data to be transformed.
        
    transformed_column : str
        The column name.
        
    ohe_path : str
        The path to the ohe_encoder object.
        
    Returns:
    -------
    set_data : pd.DataFrame
        The concatenated set data with OHE columns.
    """
    
    set_data = set_data.copy()
    
    # Load the ohe_encoder.
    ohe_encoder = joblib.load(ohe_path)
    
    # Transform the data.
    X_stasiun = np.array(set_data[transformed_column]).reshape(-1, 1)
    stasiun_features = ohe_encoder.transform(X_stasiun)
    
    # Convert to dataframe.    
    stasiun_features = pd.DataFrame(stasiun_features.tolist(), 
                                    columns = list(ohe_encoder.categories_[0]))
    
    # Set index by original set data index.
    stasiun_features.set_index(set_data.index, inplace=True)
    
    # Concatenante the new features with the original set data.
    set_data = pd.concat([stasiun_features, set_data], axis=1)
    
    # Drop the "stasiun" column.
    set_data.drop(columns="stasiun", inplace=True)
    
    # Convert columns type to string.
    new_col = [str(col_name) for col_name in set_data.columns.tolist()]
    set_data.columns = new_col
    
    # Return the feature engineered data.
    return set_data


def fit_scaler(X_concat):
    """
    Fit the scaler
    
    Parameters:
    ----------
    X_concat : pd.DataFrame
        Input data (all features must be in numeric form)
        
    Returns:
    -------
    scaler : sklearn object
        Fitted scaler object (storing the mean & std of all features)
    """
    
    scaler = StandardScaler()
    scaler.fit(X_concat)

    # Serialize the ohe_encoder object.    
    joblib.dump(scaler, '../models/scaler.pkl')
    
    return scaler

def transform_scaler(X_concat, scaler):
    """
    Transform the data using scaler
    
    Parameters:
    ----------
    X_concat : pd.DataFrame
        Input data (all features must be in numeric form)
        
    scaler : sklearn object
        Fitted scaler object (storing the mean & std of all features)
        
    Returns:
    -------
    X_concat_scaled : pd.DataFrame
        Scaled data
    """
    
    X_concat = X_concat.copy()
    
    # Transform the data
    X_concat_scaled = pd.DataFrame(
        scaler.transform(X_concat),
        columns = X_concat.columns,
        index = X_concat.index
    )
    
    return X_concat_scaled


def fit_le_encoder(y_categori):
    """
    Fit the LE encoder
    
    Parameters:
    ----------
    y_categori : pd.Series
        Categorical input label
        
    Returns:
    -------
    le_encoder : sklearn object
        Fitted LE encoder object
    """
    
    le_encoder = LabelEncoder()
    le_encoder.fit(y_categori)

    # Serialize the ohe_encoder object.    
    joblib.dump(le_encoder, '../models/le_encoder.pkl')
    
    return le_encoder

def transform_le_encoder(y_categori, le_encoder):
    """
    Transform the categorical input label using LE encoder
    
    Parameters:
    ----------
    y_categori : pd.Series
        Categorical input label
        
    le_encoder : sklearn object
        Fitted LE encoder object
        
    Returns:
    -------
    y_categori_encoded : pd.DataFrame
        Encoded categorical input label
    """
    
    y_categori = y_categori.copy()
    
    # Transform the data
    y_categori_encoded = pd.Series(
        le_encoder.transform(y_categori),        
    )
    
    return y_categori_encoded




def preprocessing(data,config,is_lb = False):
    data = data.copy()
    data = join_categories(data, config)

    # missing value
    data = nan_replace(data)

    # imputation
    # pm 10
    data.loc[(data['category'] == 'BAIK') & (data['pm10'].isna()), 'pm10'] = config['impute_pm10']["BAIK"]
    data.loc[(data['category'] == 'TIDAK BAIK') & (data['pm10'].isna()), 'pm10'] = config['impute_pm10']["TIDAK BAIK"]

    #pm25
    data.loc[(data['category'] == 'BAIK') & (data['pm25'].isna()), 'pm25'] = config['impute_pm25']["BAIK"]
    data.loc[(data['category'] == 'TIDAK BAIK') & (data['pm25'].isna()), 'pm25'] = config['impute_pm25']["BAIK"]

    # so2, co, o3, no2 Imputation
    impute_values = {
        'so2' : config['impute_so2'],
        'co' : config['impute_co'],
        'o3' : config['impute_o3'],
        'no2' : config['impute_no2']
    }

    data = data.fillna(value = impute_values)
    print(data.isna().sum())

    # encoding
    
    data = transform_ohe_encoder(
        set_data = data,
        transformed_column = "stasiun",
        ohe_path = config['path_ohe_stasiun']
    )

    # scalling
    PATH_SCALER = "../models/scaler.pkl"
    scaler = joblib.load(PATH_SCALER)

    target = data['category']
    data = transform_scaler(X_concat = data.drop(columns='category'),
                              scaler = scaler)
    data = pd.concat([data, target], axis=1)

    # load balancing
    if is_lb:
        # SMOTE.
        smote = SMOTE(random_state = 123)
        
        X_sm, y_sm = smote.fit_resample(data.drop('category', axis=1),
                                        data['category'])
        
        data = pd.concat([X_sm, y_sm], axis=1)

    print(data['category'].value_counts())
    
    # label encoding
    PATH_ENCODER = "../models/le_encoder.pkl"
    le_category = joblib.load(PATH_ENCODER)
    category_encoded = transform_le_encoder(data["category"], le_category)

    data["category"] = category_encoded.values.tolist()


    X = data.drop(columns = 'category')
    y = data['category']
    
    return X,y
    



def main():
    PATH_CONFIG = "../config/config.yaml"
    config = load_config(PATH_CONFIG)

    data_train,data_valid,data_test = load_data(config)
    
    X_train,y_train = preprocessing(data_train,config,True)
    print(X_train.head())
    print(y_train.head())
    
if __name__ == "__main__":
    main()