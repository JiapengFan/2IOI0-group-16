from memory_profiler import profile # import this package
from training.predictionAlgo import naiveNextEventPredictor, naiveTimeToNextEventPredictor, naiveAverageTimeOfCasePredictor
from preprocessing.dataParsing import parseData
from preprocessing.dataSplitting import dataSplitter
from training.decisionTree import dummy_trainers, x_prediction, fit_tree, tree_predict, quick_dummy
import pandas as pd
import timeit

@profile(precision=4) #place profile before the function, this will return memory use when running the function
def function():
    # Convert csv into dataframe
    df_2012 = pd.read_csv('.\data\BPI2012Training.csv')
    df_2012_Test = pd.read_csv('.\data\BPI2012Test.csv')

    # Parse data
    (df_training, df_2012_last_event_per_case) = parseData(df_2012)
    (df_test, df_2012_last_event_per_case_Test) = parseData(df_2012_Test)

    (df_training, df_validation, df_test) = dataSplitter(df_training, df_test)

    #naiveNextEventPredictor(df_2012, df_2012_Test)
    #naiveTimeToNextEventPredictor(df_2012, df_2012_Test)
    #naiveAverageTimeOfCasePredictor(df_2012_last_event_per_case)
    df_training_dummy = quick_dummy(df_training, 'event concept:name')
    df_test_dummy = quick_dummy(df_test, 'event concept:name')
    df_validation_dummy = quick_dummy(df_validation, 'event concept:name')

    # prediction using the decision tree
    X_train, y_train = dummy_trainers(df_training_dummy)  # current df_training doenst contain dummy variables yet
    X_validation = x_prediction(df_validation_dummy)
    X_test = x_prediction(df_test_dummy)
    decision_boom = fit_tree(X_train, y_train)
    df_Predictions = tree_predict(X_test, df_test, decision_boom)


starttime = timeit.default_timer()
print("The start time is :",starttime)
function()
print("The time difference is :", timeit.default_timer() - starttime)