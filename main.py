from functions.simple_testing import confusion_matrix_time, confusion_matrix_event, MSEcalc
from functions.predictionAlgo import naiveNextEventPredictor, naiveTimeToNextEventPredictor, naiveAverageTimeOfCasePredictor
from functions.dataParsing import parseData
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# Convert csv into dataframe
df_2012 = pd.read_csv('.\data\BPI2012Training.csv')
df_2012_Test = pd.read_csv('.\data\BPI2012Test.csv')

# Parse data
(df_2012, df_2012_last_event_per_case) = parseData(df_2012)
(df_2012_Test, df_2012_last_event_per_case_Test) = parseData(df_2012_Test)

# Invoking predictors, see naming of functions for their purposes
(dfPredictedEvent, uniqueEvents) = naiveNextEventPredictor(df_2012, df_2012_Test)
# dfPredictedTimePerCase = naiveAverageTimeOfCasePredictor(df_2012_last_event_per_case, df_2012_last_event_per_case_Test)
# dfPredictedTime = naiveTimeToNextEventPredictor(df_2012, df_2012_Test)

# Validation process that returns its accuracy or a df in confusion matrix format
# mse_predicted_time_per_case = MSEcalc(dfPredictedTimePerCase, 'unix_rel_event_time', 'predictedTime') # Giving weird errors
confusion_matrix_event(dfPredictedEvent, 'actualNextEvent', 'predictedNextEvent', uniqueEvents).to_pickle('confusion_matrix_event.pkl')  # where to save it, usually as a .pkl
# Commented out because of missing column with event's actual time
# confusion_matrix_time(dfPredictedTime, 'actualNextEvent', 'predictedNextEvent', dfPredictedTime['unix_rel_event_time'].std()).to_pickle('confusion_matrix_time.pkl')

# Read stored confusion matrices
confusion_matrix_event = pd.read_pickle('confusion_matrix_event.pkl')
# See reason why above is commented out
# confusion_matrix_time = pd.read_pickle('confusion_matrix_time.pkl')

# # Plot confusion matrix
# cols = confusion_matrix_event.columns
# confusion_matrix = pd.crosstab(confusion_matrix_event[cols], confusion_matrix_event[cols])
# sn.heatmap(confusion_matrix)
# plt.show()