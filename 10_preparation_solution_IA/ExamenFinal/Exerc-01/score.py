import preprocessing as pf
import config
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib
# =========== scoring pipeline =========

# impute categorical variables
def predict(data):
    
    # Extract first letter from cabin
    data['cabin'] = data['cabin'].str[0]

    # Impute NA categorical variables
    for var in config.CATEGORICAL_VARS:
        data[var].fillna('Missing', inplace=True)
    
    # Impute NA numerical variables
    for var in config.MISSING_INDICATOR_VARS:
        data[var+'_NA'] = np.where(data[var].isnull(), 1, 0)
        median_val = data[var].median()
        data[var].fillna(median_val, inplace=True)
    
    # Group rare labels
    for var in config.CATEGORICAL_VARS:
        frequent_ls = pf.find_frequent_labels(data, var, 0.05)
        data[var] = np.where(data[var].isin(frequent_ls), data[var], 'Rare')

    # Encode categorical variables
    for var in config.CATEGORICAL_VARS:
        data = pd.concat([data, pd.get_dummies(data[var], prefix=var, drop_first=True)], axis=1)
        data.drop(labels=var, axis=1, inplace=True)

    # Scale variables
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Make predictions using the trained model
    model = joblib.load(config.OUTPUT_MODEL_PATH)  # Load the trained model
    predictions = model.predict(data)

    return predictions


# ======================================
    
# small test that scripts are working ok
    
if __name__ == '__main__':
        
    from sklearn.metrics import accuracy_score    
    import warnings
    warnings.simplefilter(action='ignore')
    
    # Load data
    data = pf.load_data(config.PATH_TO_DATASET)
    
    X_train, X_test, y_train, y_test = pf.divide_train_test(data,
                                                            config.TARGET)
    
    pred = predict(X_test)
    
    # evaluate
    
    print('test accuracy: {}'.format(accuracy_score(y_test, pred)))
    print()
        