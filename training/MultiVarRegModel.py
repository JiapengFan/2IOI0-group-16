from training.predictionAlgo import naiveTimeToNextEventPredictor
from preprocessing.dataParsing import parseData
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from math import sqrt
import datetime as dt

def RegModel(df_training, df_test, coreFeatures):
    event_name = df_training[coreFeatures[1]].unique().tolist()

    def eventTimeConverter(inputData):
        inputData["hour"] = pd.to_datetime(inputData[coreFeatures[2]]).dt.hour
        return inputData

    print("Calculating hour of day for event and adding it to the dataset")
    df_training = eventTimeConverter(df_training)
    df_test = eventTimeConverter(df_test)

    print("Applying one hot encoding to the events")
    for x in df_training[coreFeatures[1]].unique():
        df_training[x] = 0
        df_test[x] = 0

    for x in df_training[coreFeatures[1]].unique():
        df_training[x] = np.where((df_training[coreFeatures[1]] == x), 1, 0)
        df_test[x] == np.where((df_test[coreFeatures[1]] == x), 1, 0)

    predictors = ['hour'] + event_name[:-1]

    x_train = df_training[predictors].copy()
    y_train = df_training[['actual_time_to_next_event']].copy()

    print("Fitting the training data for the linear regression model")
    model = LinearRegression().fit(x_train, y_train)
    R2 = model.score(x_train, y_train)

    x_test = df_test[predictors].copy()
    y_test = df_test[['actual_time_to_next_event']].copy()

    print("Making predictions on test dataset with the linear regression model")
    time_pred = model.predict(x_test)
    y_test['predicted_time'] = time_pred

    RMSE = sqrt(metrics.mean_squared_error(y_test['actual_time_to_next_event'], y_test['predicted_time']))
    df_test["timePrediction"] = time_pred
    return (RMSE, df_test)
