import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from IPython.display import display, HTML
from testing.testing_events import confusion_matrix_event
from training.predictionAlgo import naiveNextEventPredictor
from preprocessing.dataParsing import parseData
from preprocessing.dataSplitting import dataSplitter
from training.decisionTree import dummy_trainers, x_prediction, fit_tree, tree_predict, quick_dummy
# from streamlit import components


# Convert csv into dataframe
df_2012 = pd.read_csv('.\data\BPI2012Training.csv')
df_2012_Test = pd.read_csv('.\data\BPI2012Test.csv')

#parse data
(df_training, df_2012_last_event_per_case_train) = parseData(df_2012)
(df_test, df_2012_last_event_per_case_test) = parseData(df_2012_Test)

# Clean and split the data into train, validation & test data
(df_training, df_validation, df_test) = dataSplitter(df_training, df_test)

# Parse data
#(df_2012_train, df_2012_last_event_per_case_train) = parseData(df_training)
#(df_2012_val, df_2012_last_event_per_case_val) = parseData(df_validation)
#(df_2012_test, df_2012_last_event_per_case_test) = parseData(df_test)

# Predict next event using naive predictor
(dfPredictedEvent_training, dfPredictedEvent_test) = naiveNextEventPredictor(df_training, df_test)

#predict using the decision tree
df_training_dummy = quick_dummy(df_training, 'event concept:name')
df_test_dummy = quick_dummy(df_test, 'event concept:name')
df_validation_dummy = quick_dummy(df_validation, 'event concept:name')

#prediction using the decision tree
X_train, y_train = dummy_trainers(df_training_dummy) #current df_training doenst contain dummy variables yet
X_validation = x_prediction(df_validation_dummy)
X_test = x_prediction(df_test_dummy)
decision_boom = fit_tree(X_train, y_train)
df_Predictions = tree_predict(X_test, df_test, decision_boom)


#uniqueEvents = dfPredictedEvent_test['actual_next_event'].unique()
#actual_array = dfPredictedEvent_test['actual_next_event'].to_numpy()
uniqueEvents = df_Predictions['event concept:name'].unique()
actual_array = df_Predictions['event concept:name'][1:].to_numpy()
#predicted_array = dfPredictedEvent_test['naive_predicted_next_event'].to_numpy()
predicted_array = df_Predictions['predictedNextEvent'][1:].to_numpy()
confusion_matrix_event = confusion_matrix_event(actual_array, predicted_array, uniqueEvents)

# Plot confusion matrix
#confusion_matrix = pd.crosstab(confusion_matrix_event[uniqueEvents], confusion_matrix_event[uniqueEvents])

#confusion matrix with color
sns.heatmap(confusion_matrix_event, cmap = 'Reds')
plt.show()

#confusion matrix table
#confusion_matrix_event.to_csv('.\data\Confusion_matrix_sprint2_naive.csv')
#display(HTML(confusion_matrix_event.style.render()))


#plt.pcolor(confusion_matrix_event)
#plt.yticks(np.arange(0.5, len(confusion_matrix_event.index), 1), confusion_matrix_event.index)
#plt.xticks(np.arange(0.5, len(confusion_matrix_event.columns), 1), confusion_matrix_event.columns)
#plt.show()

#accuracy
acc = 0
for a in range(len(actual_array)):
    if actual_array[a] == predicted_array[a]:
        acc += 1

print(acc/len(actual_array))