import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


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
    
    print (X_train[config.MISSING_INDICATOR_VARS].isnull().sum())
    return X_train, X_test

    
    
#def impute_na():
def impute_na(X_train, X_test):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    #vars_cat = [c for c in data.columns if data[c].dtypes=='O']
    X_train[config.CATEGORICAL_VARS] = X_train[config.CATEGORICAL_VARS].fillna('Missing')
    X_test[config.CATEGORICAL_VARS] = X_test[config.CATEGORICAL_VARS].fillna('Missing')
    return X_train, X_test



def find_frequent_labels(df, var, rare_perc):
    
    # function finds the labels that are shared by more than
    # a certain % of the passengers in the dataset
    
    df = df.copy()
    
    tmp = df.groupby(var)[var].count() / len(df)
    
    return tmp[tmp > rare_perc].index



    
def remove_rare_labels(X_train, X_test):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    for var in config.CATEGORICAL_VARS:
    
        # find the frequent categories
        frequent_ls = find_frequent_labels(X_train, var, 0.05)
        print(var)
        print(frequent_ls)
    
        # replace rare categories by the string "Rare"
        X_train[var] = np.where(X_train[var].isin(
        frequent_ls), X_train[var], 'Rare')
    
        X_test[var] = np.where(X_test[var].isin(
        frequent_ls), X_test[var], 'Rare')

    return X_train, X_test


#def encode_categorical(df, var):
def encode_categorical(X_train, X_test):
    # adds ohe variables and removes original categorical variable
    
    #df = df.copy()
    
    for var in config.CATEGORICAL_VARS:
    
        # to create the binary variables, we use get_dummies from pandas
    
        X_train = pd.concat([X_train,
                         pd.get_dummies(X_train[var], prefix=var, drop_first=True)
                         ], axis=1)
    
        X_test = pd.concat([X_test,
                        pd.get_dummies(X_test[var], prefix=var, drop_first=True)
                        ], axis=1)
    

    X_train.drop(labels=config.CATEGORICAL_VARS, axis=1, inplace=True)
    X_test.drop(labels=config.CATEGORICAL_VARS, axis=1, inplace=True)

    print ('X_train.shape: ', X_train.shape)
    print ('X_test.shape:  ', X_test.shape)
    
    X_test['embarked_Rare'] = 0
    
    
    return X_train, X_test



def check_dummy_variables(df, dummy_list):
    
    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    pass
    

def train_scaler(df, output_path):
    # train and save scaler
    pass
   
  
    

# def scale_features(df, output_path):
def scale_features(X_train, X_test):
    # load scaler and transform data
    
    # Get the column order from the training set
    column_order = X_train.columns
    
     # create scaler
    scaler = StandardScaler()

    #  fit  the scaler to the train set
    scaler.fit(X_train[column_order]) 

    # transform the train and test set
    X_train = scaler.transform(X_train[column_order])

    X_test = scaler.transform(X_test[column_order])
    
    return X_train, X_test



#def train_model(df, target, output_path):
def train_model(X_train, y_train):
    # train and save model
    model = LogisticRegression(C=0.0005, random_state=0)

    # train the model
    model.fit(X_train, y_train)
    
    return model



#def predict(df, model):
def predict(X_train, y_train, model):
    # load model and get predictions
    # make predictions for test set
    class_ = model.predict(X_train)
    pred = model.predict_proba(X_train)[:,1]

    # determine mse and rmse
    print('train roc-auc: {}'.format(roc_auc_score(y_train, pred)))
    print('train accuracy: {}'.format(accuracy_score(y_train, class_)))
    print()

    # make predictions for test set
    class_ = model.predict(X_test)
    pred = model.predict_proba(X_test)[:,1]

    # determine mse and rmse
    print('test roc-auc: {}'.format(roc_auc_score(y_test, pred)))
    print('test accuracy: {}'.format(accuracy_score(y_test, class_)))
    print()


if __name__ == '__main__':
    df =load_data()
    #print('type(df)', type(df))
    #print ('df')
    #print(df.head(3))
    X_train, X_test, y_train, y_test =  divide_train_test(df)
    X_train_2, X_test_2 = extract_cabin_letter(X_train, X_test)
    X_train_3, X_test_3 = add_missing_indicator(X_train_2, X_test_2)
    X_train_4, X_test_4 = impute_na(X_train_3, X_test_3)
    X_train_5, X_test_5 = remove_rare_labels(X_train_4, X_test_4)
    X_train_6, X_test_6 = encode_categorical(X_train_5, X_test_5)
    print('X_train_6: ', X_train_6.head(3))
    print('X_test_6: ', X_test_6.head(3))
    print('X_train_columns: ', X_train_6.columns)
    print('X_test_columns: ', X_test_6.columns)
    X_train_7, X_test_7 = scale_features(X_train_6, X_test_6)
    model = train_model(X_train_7, y_train)
    predict(X_train_7, y_train, model)