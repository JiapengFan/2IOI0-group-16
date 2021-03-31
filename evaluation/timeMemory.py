from memory_profiler import profile # import this package
from training.predictionAlgo import naiveNextEventPredictor, naiveTimeToNextEventPredictor, naiveAverageTimeOfCasePredictor
from preprocessing.dataParsing import parseData
from preprocessing.dataSplitting import dataSplitter
from training.decisionTree import dummy_trainers, x_prediction, fit_tree, tree_predict, quick_dummy
import pandas as pd
import timeit
import os
import pandas as pd
import numpy as np
from preprocessing.dataParsing import parseData
from preprocessing.dataSplitting import dataSplitter
from training.predictionAlgo import naiveNextEventPredictor, naiveTimeToNextEventPredictor
from training.MultiVarRegModel import RegModel
import tkinter as tk
from tkinter import *
from LSTM.lstmPrediction import LSTMEvent, LSTMTime
import warnings
from training.RandomForestPredictor import run_full_rf
import os

warnings.filterwarnings("ignore")

dirname = os.path.dirname(__file__)

@profile(precision=4) #place profile before the function, this will return memory use when running the function
def function():


    loadEpoch=""
    base_features = ["case concept:name", "event concept:name", "event time:timestamp"]
    extra_features = []

    # Convert csv into dataframe
    print("Loading datasets")
    df_training_raw = pd.read_csv(dirname + "/data/training.csv")
    df_test_raw = pd.read_csv(dirname + "/data/test.csv")
    df_test_columns = df_test_raw.columns
    # Parsing data
    print("Parsing and splitting the data")
    (df_training, df_2012_last_event_per_case_train) = parseData(df_training_raw)
    (df_test, df_2012_last_event_per_case_test) = parseData(df_test_raw)

    # Clean and split the data into train, validation & test data
    (df_training, df_validation, df_test) = dataSplitter(df_training, df_test)

    # Apply the naive predictors to all the datasets
    print("Apply naive predictor and find actual next event and time to next event, as well as generating naive predictions")
    (df_training, df_test) = naiveTimeToNextEventPredictor(df_training, df_test)
    (df_training, df_test) = naiveNextEventPredictor(df_training, df_test)
    (df_training, df_validation) = naiveTimeToNextEventPredictor(df_training, df_validation)
    (df_training, df_validation) = naiveNextEventPredictor(df_training, df_validation)

    # Run prediction algorithms associated with input
    accuracy, df_test = run_full_rf(df_training, df_test, base_features)
    accuracy, df_test = LSTMEvent(df_training, df_validation, df_test, base_features, extra_features, int(10))
    RMSE, df_test = LSTMTime(df_training, df_validation, df_test, base_features, extra_features, int(10))
    RMSE, df_test = RegModel(df_training, df_test, base_features)

    for x in df_test.columns:
        if (x not in df_test_columns):
            if not (x == "timePrediction" or x == "eventPrediction" or x == "naive_predicted_time_to_next_event" or x == "naive_predicted_next_event"):
                df_test.drop(columns=x, inplace=True)

    #if ("naive" in time_pred.get()):
        #df_test.drop(columns="naive_predicted_time_to_next_event", inplace=True)
    #if ("naive" in event_pred.get()):
        #df_test.drop(columns="naive_predicted_next_event", inplace=True)

    print("Outputting csv file")
    print(df_test.head(10))
    df_test.to_csv(dirname + "/output/output.csv", index=False)
    print("CSV file is outputted to: ", dirname + "/output/output.csv")
    print("Finished processing request!!")


starttime = timeit.default_timer()
print("The start time is :",starttime)
function()
print("The time difference is :", timeit.default_timer() - starttime)