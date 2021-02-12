from functions.simple_testing import confusion_matrix_time, confusion_matrix_event, MSEcalc
from functions.predictionAlgo import naiveNextEventPredictor, naiveTimeToNextEventPredictor, naiveAverageTimeOfCasePredictor
from functions.dataParsing import parseData
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

mse_predicted_time_per_case = MSEcalc(dfPredictedTimePerCase, 'unix_rel_event_time', 'predictedTime')
confusion_matrix_event(dfPredictedEvent, 'actualNextEvent', 'timeToNextEvent', uniqueEvents).to_pickle('confusion_matrix_event.pkl')  # where to save it, usually as a .pkl
confusion_matrix_time(dfPredictedTime, 'actualNextEvent', 'predictedNextEvent', dfPredictedTime['unix_rel_event_time'].std()).to_pickle('confusion_matrix_time.pkl')

confusion_matrix_event = pd.read_pickle('confusion_matrix_event.pkl')
confusion_matrix_time = pd.read_pickle('confusion_matrix_time.pkl')

print(mse_predicted_time_per_case, '\n')
print(confusion_matrix_event, '\n')
print(confusion_matrix_time, '\n')