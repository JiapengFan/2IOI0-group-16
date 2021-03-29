from training.predictionAlgo import naiveNextEventPredictor, naiveTimeToNextEventPredictor, naiveAverageTimeOfCasePredictor
from preprocessing.dataParsing import parseData
from preprocessing.dataSplitting import dataSplitter
import pandas as pd

from sklearn import metrics
from math import sqrt
import numpy as np

from researchPaperPrediction import timePrediction
from training.lstmTimeToNextEventPredictor import lstmTimePredictor


# Convert csv into dataframe
df_2012 = pd.read_csv("BPI2012Training.csv")
df_2012_Test = pd.read_csv("BPI2012Test.csv")

# Parse data
(df_training, df_2012_last_event_per_case) = parseData(df_2012)
(df_test, df_2012_last_event_per_caquitse_Test) = parseData(df_2012_Test)

(df_training, df_validation, df_test) = dataSplitter(df_training, df_test)

(df_training, df_test) = naiveTimeToNextEventPredictor(df_training, df_test)
(df_training, df_test) = naiveNextEventPredictor(df_training, df_test)
(df_training, df_validation) = naiveTimeToNextEventPredictor(df_training, df_validation)
(df_training, df_validation) = naiveNextEventPredictor(df_training, df_validation)

RMSE, df_test = lstmTimePredictor(df_training, df_validation, df_test, [
                                  "case concept:name", "event concept:name", "event time:timestamp"], [])

print(RMSE)