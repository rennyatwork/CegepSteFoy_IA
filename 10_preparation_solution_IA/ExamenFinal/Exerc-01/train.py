import joblib
import preprocessing as pf
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import config

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data


# divide data set



# get first letter from cabin variable



# impute categorical variables



# impute numerical variable



# Group rare labels



# encode categorical variables



# check all dummies were added



# train scaler and save



# scale train set



# train model and save
def train_model(X_train, y_train):
    # train and save model
    model = LogisticRegression(C=0.0005, random_state=0)

    # train the model
    model.fit(X_train, y_train)
    joblib.dump(model, config.OUTPUT_MODEL_PATH)
    return model

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
    df =pf.load_data()
    #print('type(df)', type(df))
    #print ('df')
    #print(df.head(3))
    X_train, X_test, y_train, y_test =  pf.divide_train_test(df)
    X_train_2, X_test_2 = pf.extract_cabin_letter(X_train, X_test)
    X_train_3, X_test_3 = pf.add_missing_indicator(X_train_2, X_test_2)
    X_train_4, X_test_4 = pf.impute_na(X_train_3, X_test_3)
    X_train_5, X_test_5 = pf.remove_rare_labels(X_train_4, X_test_4)
    X_train_6, X_test_6 = pf.encode_categorical(X_train_5, X_test_5)
    print('X_train_6: ', X_train_6.head(3))
    print('X_test_6: ', X_test_6.head(3))
    print('X_train_columns: ', X_train_6.columns)
    print('X_test_columns: ', X_test_6.columns)
    X_train_7, X_test_7 = pf.scale_features(X_train_6, X_test_6)
    model = train_model(X_train_7, y_train)
    predict(X_train_7, y_train, model)

    print('Finished training')