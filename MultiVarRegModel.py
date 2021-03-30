from training.predictionAlgo import naiveTimeToNextEventPredictor
from preprocessing.dataParsing import parseData
import pandas as pd

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from math import sqrt
import datetime as dt

# Pands option to disable scientific notation and display numeric values with 3 decimals
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Convert csv into dataframe
df_training_raw = pd.read_csv('.\data\BPI2012Training.csv')
df_test_raw = pd.read_csv('.\data\BPI2012Test.csv')

#parsing data
(df_training, df_2012_last_event_per_case_train) = parseData(df_training_raw)
(df_test, df_2012_last_event_per_case_test) = parseData(df_test_raw)

(df_training, df_test) = naiveTimeToNextEventPredictor(df_training, df_test)

def RegModel(df_training, df_test):
    event_name = df_training['event concept:name'].unique().tolist()

    def eventTimeConverter(inputData):

        inputData["hour"] = pd.to_datetime(inputData['event time:timestamp']).dt.hour

        return inputData

    df_training = eventTimeConverter(df_training)
    df_test = eventTimeConverter(df_test)

    def oneHotEncoding(inputData, attr):

        one_hot = pd.get_dummies(inputData[attr])
        df = inputData.join(one_hot)

        return (df)

    df_training = oneHotEncoding(df_training, 'event concept:name')
    df_test = oneHotEncoding(df_test, 'event concept:name')

    predictors = ['case AMOUNT_REQ', 'hour'] + event_name[:-1]

    x_train = df_training[predictors].copy()
    y_train = df_training[['actual_time_to_next_event']].copy()

    model = LinearRegression().fit(x_train, y_train)
    R2 = model.score(x_train, y_train)

    x_test = df_test[predictors].copy()
    y_test = df_test[['actual_time_to_next_event']].copy()

    time_pred = model.predict(x_test)
    y_test['predicted_time'] = time_pred

    RMSE = sqrt(metrics.mean_squared_error(y_test['actual_time_to_next_event'], y_test['predicted_time']))

    return (y_test, R2, RMSE)
