import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib

import config


# Individual pre-processing and training functions
# ================================================

def load_data(df_path=config.PATH_TO_DATASET):
    # Function loads data for training
    data = pd.read_csv(df_path)
    # print('data: ', data.head(3))
    print('cols: ',len(data.columns))
    return data


def divide_train_test(df, target=config.TARGET):
    # Function divides data set in train and test
    
    data=df
    print('type: ', type(data))
    X_train, X_test, y_train, y_test = train_test_split(
    data.drop(target, axis=1),  # predictors
    data[target],  # target
    test_size=0.2,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility


    print('X_train.shape: ', X_train.shape) 
    print('X_test: ', X_test.shape)
    
    return X_train, X_test, y_train, y_test



#def extract_cabin_letter(df, var):
def extract_cabin_letter(X_train, X_test, var_cabin='cabin'):
    # captures the first letter
    
    X_train[var_cabin] = X_train[var_cabin].str[0] # captures the first letter
    X_test[var_cabin] = X_test[var_cabin].str[0] # captures the first letter

    unique_vals = X_train[var_cabin].unique()
    print ("[extract_cabin_letter], unique_vals: ", unique_vals)
    
    return X_train, X_test



# def add_missing_indicator(df, var):
def add_missing_indicator(X_train, X_test):
    for var in config.MISSING_INDICATOR_VARS:
        print ("[var]: ", var)
        # add missing indicator
        X_train[var+'_NA'] = np.where(X_train[var].isnull(), 1, 0)
        X_test[var+'_NA'] = np.where(X_test[var].isnull(), 1, 0)

        # replace NaN by median
        median_val = X_train[var].median()
        print(var, median_val)

        X_train[var].fillna(median_val, inplace=True)
        X_test[var].fillna(median_val, inplace=True)
    
    print (X_train[['age', 'fare']].isnull().sum())
    return X_train, X_test

    
    
    


    
def impute_na():
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    pass



def remove_rare_labels():
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    pass



def encode_categorical(df, var):
    # adds ohe variables and removes original categorical variable
    
    df = df.copy()
    
    pass



def check_dummy_variables(df, dummy_list):
    
    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    pass
    

def train_scaler(df, output_path):
    # train and save scaler
    pass
  
    

def scale_features(df, output_path):
    # load scaler and transform data
    pass



def train_model(df, target, output_path):
    # train and save model
    pass



def predict(df, model):
    # load model and get predictions
    pass


if __name__ == '__main__':
    df =load_data()
    #print('type(df)', type(df))
    #print ('df')
    #print(df.head(3))
    X_train, X_test, y_train, y_test =  divide_train_test(df)
    X_train_2, X_test_2 = extract_cabin_letter(X_train, X_test)
    X_train_3, X_test_3 = add_missing_indicator(X_train_2, X_test_2)