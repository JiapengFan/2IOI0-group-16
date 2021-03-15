from training.predictionAlgo import naiveNextEventPredictor, naiveTimeToNextEventPredictor, naiveAverageTimeOfCasePredictor
from preprocessing.dataParsing import parseData
from preprocessing.dataSplitting import dataSplitter
import pandas as pd

from sklearn import metrics
from math import sqrt
import numpy as np

from researchPaperPrediction import timePrediction


# Convert csv into dataframe
df_2012 = pd.read_csv('.\data\BPI2012Training.csv')
df_2012_Test = pd.read_csv('.\data\BPI2012Test.csv')

# Parse data
(df_training, df_2012_last_event_per_case) = parseData(df_2012)
(df_test, df_2012_last_event_per_case_Test) = parseData(df_2012_Test)

(df_training, df_validation, df_test) = dataSplitter(df_training, df_test)

(df_training, df_test) = naiveTimeToNextEventPredictor(df_training, df_test)
(df_training, df_test) = naiveNextEventPredictor(df_training, df_test)


(df_training, df_validation, df_test) = timePrediction(
    df_training, df_validation, df_test)


print(df_test.head(50))

df_test["research"] = np.where(
    (np.isnan(df_test['research'])), 0, df_test["research"])

df_test["actual_time_to_next_event"] = np.where((np.isnan(
    df_test["actual_time_to_next_event"])), 0, df_test["actual_time_to_next_event"])

RMSE = sqrt(metrics.mean_squared_error(
    df_test[df_test['event concept:name'] != "W_Nabellen offertes"]["actual_time_to_next_event"], df_test[df_test['event concept:name'] != "W_Nabellen offertes"]["research"]))


print(RMSE)

RMSE = sqrt(metrics.mean_squared_error(
    df_test[df_test['event concept:name'] != "W_Nabellen offertes"]["actual_time_to_next_event"], df_test[df_test['event concept:name'] != "W_Nabellen offertes"]["naive_predicted_time_to_next_event"]))

print(RMSE)
