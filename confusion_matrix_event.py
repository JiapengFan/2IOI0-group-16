import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from testing.testing_events import confusion_matrix_event
from training.predictionAlgo import naiveNextEventPredictor
from preprocessing.dataParsing import parseData

# Convert csv into dataframe
df_2012 = pd.read_csv('.\data\BPI2012Training.csv')
df_2012_Test = pd.read_csv('.\data\BPI2012Test.csv')

# Parse data
(df_2012, df_2012_last_event_per_case) = parseData(df_2012)
(df_2012_Test, df_2012_last_event_per_case_Test) = parseData(df_2012_Test)

(dfPredictedEvent, uniqueEvents) = naiveNextEventPredictor(df_2012, df_2012_Test)

uniqueEvents.append('nan')
actual_array = dfPredictedEvent['actualNextEvent'].to_numpy()
predicted_array = dfPredictedEvent['predictedNextEvent'].to_numpy()
confusion_matrix_event = confusion_matrix_event(actual_array, predicted_array, uniqueEvents)

# Plot confusion matrix
cols = confusion_matrix_event.columns
confusion_matrix = pd.crosstab(confusion_matrix_event[cols], confusion_matrix_event[cols])
sn.heatmap(confusion_matrix)
plt.show()