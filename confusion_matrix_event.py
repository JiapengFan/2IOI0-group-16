import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from testing.testing_events import confusion_matrix_event
from training.predictionAlgo import naiveNextEventPredictor
from preprocessing.dataParsing import parseData
from preprocessing.dataSplitting import dataSplitter


# Convert csv into dataframe
df_2012 = pd.read_csv('.\data\BPI2012Training.csv')
df_2012_Test = pd.read_csv('.\data\BPI2012Test.csv')

#parse data
(df_training, df_2012_last_event_per_case_train) = parseData(df_training_raw)
(df_test, df_2012_last_event_per_case_test) = parseData(df_test_raw)

# Clean and split the data into train, validation & test data
(df_training, df_validation, df_test) = dataSplitter(df_training, df_test)

# Parse data
#(df_2012_train, df_2012_last_event_per_case_train) = parseData(df_training)
#(df_2012_val, df_2012_last_event_per_case_val) = parseData(df_validation)
#(df_2012_test, df_2012_last_event_per_case_test) = parseData(df_test)

# Predict next event using naive predictor
(dfPredictedEvent_training, dfPredictedEvent_test) = naiveNextEventPredictor(df_2012_train, df_2012_Test)

uniqueEvents = dfPredictedEvent_test['actual_next_event'].unique()
actual_array = dfPredictedEvent_test['actual_next_event'].to_numpy()
predicted_array = dfPredictedEvent_test['naive_predicted_next_event'].to_numpy()
confusion_matrix_event = confusion_matrix_event(actual_array, predicted_array, uniqueEvents)

# Plot confusion matrix
confusion_matrix = pd.crosstab(confusion_matrix_event[uniqueEvents], confusion_matrix_event[uniqueEvents])
sn.heatmap(confusion_matrix)
plt.show()