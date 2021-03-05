from testing.testing_events import confusion_matrix_time, confusion_matrix_event, MSEcalc
from training.predictionAlgo import naiveNextEventPredictor, naiveTimeToNextEventPredictor, naiveAverageTimeOfCasePredictor
from preprocessing.dataParsing import parseData
import pandas as pd

# Convert csv into dataframe
df_2012 = pd.read_csv('.\data\BPI2012Training.csv')
df_2012_Test = pd.read_csv('.\data\BPI2012Test.csv')

# Parse data
(df_2012, df_2012_last_event_per_case) = parseData(df_2012)
(df_2012_Test, df_2012_last_event_per_case_Test) = parseData(df_2012_Test)

(dfPredictedEvent, uniqueEvents) = naiveNextEventPredictor(df_2012, df_2012_Test)
dfPredictedTimePerCase = naiveAverageTimeOfCasePredictor(df_2012_last_event_per_case, df_2012_last_event_per_case_Test)
dfPredictedTime = naiveTimeToNextEventPredictor(df_2012, df_2012_Test)
