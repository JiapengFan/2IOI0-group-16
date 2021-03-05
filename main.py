from functions.simple_testing import confusion_matrix_time, confusion_matrix_event, MSEcalc
from functions.predictionAlgo import naiveNextEventPredictor, naiveTimeToNextEventPredictor, naiveAverageTimeOfCasePredictor
from functions.dataParsing import parseData
from functions.dataSplitting import dataSplitter
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


# Pands option to disable scientific notation and display numeric values with 3 decimals
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Convert csv into dataframe
df_training_raw = pd.read_csv('.\data\BPI2012Training.csv')
df_test_raw = pd.read_csv('.\data\BPI2012Test.csv')

# Parse data
(df_training_parsed, df_training_lastevent_parsed) = parseData(df_training_raw)
(df_test_parsed, df_test_lastevent_parsed) = parseData(df_test_raw)

# Clean and split the data into train, validation & test data
(df_training, df_validation, df_test) = dataSplitter(
    df_training_parsed, df_test_parsed)


# Invoking predictors, see naming of functions for their purposes
(df_training, df_test) = naiveNextEventPredictor(df_training, df_test)
(df_training, df_test) = naiveTimeToNextEventPredictor(df_training, df_test)

# dfPredictedTimePerCase = naiveAverageTimeOfCasePredictor(df_training_last_event_per_case, df_test_lastevent_parsed)
# dfPredictedTime = naiveTimeToNextEventPredictor(df_training, df_test)

#prediction using the decision tree
X_train, y_train = dummy_trainers(df_training) #current df_training doenst contain dummy variables yet
X_validation = x_prediction(df_validation)
X_test = x_prediction(data_test)
decision_tree = fit_tree(X_train, y_train)
df_Predictions = tree_predict(X_test, df_test, decision_tree)

# Validation process that returns its accuracy or a df in confusion matrix format
# mse_predicted_time_per_case = MSEcalc(dfPredictedTimePerCase, 'unix_rel_event_time', 'predictedTime') # Giving weird errors
confusion_matrix_event(dfPredictedEvent, 'actualNextEvent', 'predictedNextEvent', uniqueEvents).to_pickle(
    'confusion_matrix_event.pkl')  # where to save it, usually as a .pkl
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
