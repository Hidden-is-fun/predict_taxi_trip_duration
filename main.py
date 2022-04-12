import datetime

import joblib
import pandas as pd
import numpy as np
import xgboost

import taxi_utils


def predict(data):
    print(data)
    lr = joblib.load('model.model')
    res = lr.predict(xgboost.DMatrix(data))
    # res = lr.predict(xgboost.DMatrix(data))
    return res
    return np.exp(res[0]) - 1


if __name__ == "__main__":
    dt = datetime.datetime(2016, 6, 1, 11, 48, 41)
    dt64 = np.datetime64(dt)
    print(dt64)
    predict([])
    exit(0)

    TRAIN_DIR = "data/train.csv"
    TEST_DIR = "data/test.csv"

    data_train, data_test = taxi_utils.read_data(TRAIN_DIR, TEST_DIR)

    X_train = data_train.copy()
    X_test = data_test.copy()

    X_test.loc[:, 'pickup_year'] = X_test['pickup_datetime'].dt.year
    X_train.loc[:, 'pickup_year'] = X_train['pickup_datetime'].dt.year

    X_test.loc[:, 'pickup_month'] = X_test['pickup_datetime'].dt.month
    X_train.loc[:, 'pickup_month'] = X_train['pickup_datetime'].dt.month

    X_test.loc[:, 'pickup_day'] = X_test['pickup_datetime'].dt.day
    X_train.loc[:, 'pickup_day'] = X_train['pickup_datetime'].dt.day

    X_test.loc[:, 'pickup_hour'] = X_test['pickup_datetime'].dt.hour
    X_train.loc[:, 'pickup_hour'] = X_train['pickup_datetime'].dt.hour

    X_test.loc[:, 'pickup_minute'] = X_test['pickup_datetime'].dt.minute
    X_train.loc[:, 'pickup_minute'] = X_train['pickup_datetime'].dt.minute

    X_test.loc[:, 'pickup_second'] = X_test['pickup_datetime'].dt.second
    X_train.loc[:, 'pickup_second'] = X_train['pickup_datetime'].dt.second

    X_test = X_test.drop(['pickup_datetime'], axis=1)
    X_train = X_train.drop(['pickup_datetime'], axis=1)

    X_train = X_train.set_index(['id'])
    X_test = X_test.set_index(['id'])
    pd.set_option('display.width', 300)
    pd.set_option('max_column', 100)

    labels = np.log(X_train['trip_duration'].values + 1)
    X_train = X_train.drop(['trip_duration'], axis=1)

    print(X_test.values[0])

    model = taxi_utils.train_xgb(X_train, labels)
    joblib.dump(model, 'model.model')

    submission = taxi_utils.predict_xgb(model, X_test)
    print(submission.head(5))

    feature_names = X_train.columns.values
    ft_importances = taxi_utils.feature_importances(model, feature_names)
    print(ft_importances)

