#from testing.testing_events import confusion_matrix_time, confusion_matrix_event, MSEcalc
from training.predictionAlgo import naiveNextEventPredictor, naiveTimeToNextEventPredictor, naiveAverageTimeOfCasePredictor
from preprocessing.dataParsing import parseData
from preprocessing.dataSplitting import dataSplitter
#from training.predictionAlgo import dummy_trainers, x_prediction, fit_tree, tree_predict, quick_dummy
from training.decisionTree import dummy_trainers, x_prediction, fit_tree, tree_predict, quick_dummy
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


# Pands option to disable scientific notation and display numeric values with 3 decimals
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Convert csv into dataframe
df_training_raw = pd.read_csv('.\data\BPI2012Training.csv')
df_test_raw = pd.read_csv('.\data\BPI2012Test.csv')

# Clean and split the data into train, validation & test data
(df_training, df_validation, df_test) = dataSplitter(df_training_raw, df_test_raw)

# Parse data
(df_2012_train, df_2012_last_event_per_case_train) = parseData(df_training)
(df_2012_val, df_2012_last_event_per_case_val) = parseData(df_validation)
(df_2012_test, df_2012_last_event_per_case_test) = parseData(df_test)

# Invoking predictors, see naming of functions for their purposes
(df_training, df_test) = naiveNextEventPredictor(df_training, df_test)
(df_training, df_test) = naiveTimeToNextEventPredictor(df_training, df_test)

# dfPredictedTimePerCase = naiveAverageTimeOfCasePredictor(df_training_last_event_per_case, df_test_lastevent_parsed)
# dfPredictedTime = naiveTimeToNextEventPredictor(df_training, df_test)

#creating the dummy variables dataframes
df_training_dummy = quick_dummy(df_training, 'event concept:name')
df_test_dummy = quick_dummy(df_test, 'event concept:name')
df_validation_dummy = quick_dummy(df_validation, 'event concept:name')

#prediction using the decision tree
X_train, y_train = dummy_trainers(df_training_dummy) #current df_training doenst contain dummy variables yet
X_validation = x_prediction(df_validation_dummy)
X_test = x_prediction(df_test_dummy)
decision_tree = fit_tree(X_train, y_train)
df_Predictions = tree_predict(X_test, df_test, decision_tree)

print(df_Predictions)

