import pandas as pd
import numpy as np
from preprocessing.dataParsing import parseData
from preprocessing.dataSplitting import dataSplitter
from training.predictionAlgo import naiveNextEventPredictor, naiveTimeToNextEventPredictor
from training.MultiVarRegModel import RegModel
import tkinter as tk
from tkinter import *
from training.lstmPrediction import LSTMEvent, LSTMTime
import warnings
from training.RandomForestPredictor import run_full_rf
import os

warnings.filterwarnings("ignore")

dirname = os.path.dirname(os.path.abspath(__file__))
base_model = ""
epochs = 10

def setStates(*args):
    
    if ("and" in base_model.get()):
        d2.configure(state="normal")
        d3.configure(state="normal")
        if ("LSTM" in time_pred.get() or "LSTM" in event_pred.get()):
            e6.configure(state="normal")
            e7.configure(state="normal")
            e8.configure(state="normal")
            e9.configure(state="normal")
            checkboxPreviousModelLoader.configure(state="normal")
            checkboxTrainModelLoader.configure(state="normal")
        else:
            e6.configure(state="disabled")
            e7.configure(state="disabled")
            e8.configure(state="disabled")
            e9.configure(state="disabled")
            checkboxPreviousModelLoader.configure(state="disabled")
            checkboxTrainModelLoader.configure(state="disabled")
    elif ("time" in base_model.get()):
        d3.configure(state="disabled")
        d2.configure(state="normal")
        if ("LSTM" in time_pred.get()):
            e6.configure(state="normal")
            e7.configure(state="normal")
            e8.configure(state="normal")
            e9.configure(state="normal")
            checkboxPreviousModelLoader.configure(state="normal")
            checkboxTrainModelLoader.configure(state="normal")
        else:
            e6.configure(state="disabled")
            e7.configure(state="disabled")
            e8.configure(state="disabled")
            e9.configure(state="disabled")
            checkboxPreviousModelLoader.configure(state="disabled")
            checkboxTrainModelLoader.configure(state="disabled")
    elif ("event" in base_model.get()):
        d2.configure(state="disabled")
        d3.configure(state="normal")
        if ("LSTM" in event_pred.get()):
            e6.configure(state="normal")
            e7.configure(state="normal")
            e8.configure(state="normal")
            e9.configure(state="normal")
            checkboxPreviousModelLoader.configure(state="normal")
            checkboxTrainModelLoader.configure(state="normal")
        else:
            e6.configure(state="disabled")
            e7.configure(state="disabled")
            e8.configure(state="disabled")
            e9.configure(state="disabled")
            checkboxPreviousModelLoader.configure(state="disabled")
            checkboxTrainModelLoader.configure(state="disabled")

    if (loadPreviousModels.get() == 1):
        checkboxPreviousModelLoader.configure(state="normal")
        checkboxTrainModelLoader.configure(state="disabled")    
    elif (trainPreviousModels.get() == 1):
        checkboxTrainModelLoader.configure(state="normal")
        checkboxPreviousModelLoader.configure(state="disabled")
    elif  ("LSTM" in event_pred.get() or "LSTM" in time_pred.get()):
        checkboxTrainModelLoader.configure(state="normal")
        checkboxPreviousModelLoader.configure(state="normal")
    
    #if ()

    if (loadPreviousModels.get() == 1 and ("LSTM" in event_pred.get() or "LSTM" in time_pred.get())):
        e13.configure(state="normal")
    else:
        e13.configure(state="disabled")

    if (trainPreviousModels.get() == 1 and ("LSTM" in event_pred.get() or "LSTM" in time_pred.get())):
        e14.configure(state="normal")
    else:
        e14.configure(state="disabled")



master = tk.Tk()
master.title("2IOI0 DBL Process mining group 16")
e1 = tk.Entry(master)
e2 = tk.Entry(master)
e3 = tk.Entry(master)
e6 = tk.Entry(master)
e7 = tk.Entry(master)
e8 = tk.Entry(master)
e9 = tk.Entry(master)
e10 = tk.Entry(master)
e11 = tk.Entry(master)
e12 = tk.Entry(master)
e13 = tk.Entry(master)
e14 = tk.Entry(master)
e15 = tk.Entry(master)
e16 = tk.Entry(master)
tk.Label(master, text='Time prediction').grid(row=7, column=0)
time_pred = StringVar(master)
time_pred.set("LSTM")
time_pred.trace("w", setStates)
d2 = OptionMenu(master, time_pred, 'LSTM', 'Multivariate regression', 'naive predictor')
d2.grid(row=7, column=1)
tk.Label(master, text='Event prediction').grid(row=7, column=2)
event_pred = StringVar(master)
event_pred.set("LSTM")
event_pred.trace("w", setStates)
d3 = OptionMenu(master, event_pred, 'LSTM', 'Random forest', 'naive predictor')
d3.grid(row=7, column=3)
tk.Label(master, text="").grid(row=12)
tk.Label(master, text='Model selection').grid(row=3)
base_model = StringVar(master)
base_model.set("time and event prediction")
base_model.trace("w", setStates)
d1 = OptionMenu(master, base_model, 'event prediction', 'time prediction', 'time and event prediction')
d1.grid(row=3, column=1)
loadPreviousModels = tk.IntVar()
loadPreviousModels.trace("w", setStates)
checkboxPreviousModelLoader = tk.Checkbutton(master, text="Load previous LSTM model", variable=loadPreviousModels, onvalue=1, offvalue=0, )
trainPreviousModels = tk.IntVar()
trainPreviousModels.trace("w", setStates)
checkboxTrainModelLoader = tk.Checkbutton(master, text="Train previous LSTM model", variable=trainPreviousModels, onvalue=1, offvalue=0, )
checkboxPreviousModelLoader.grid(row=13, column = 0)
checkboxTrainModelLoader.grid(row=13, column = 2)
e13.configure(state="disabled")
e14.configure(state="disabled")
e15.configure(state="disabled")
e16.configure(state="disabled")

def User_inputGUI():

    tk.Label(master, text="Column names of the following core features").grid(row=14)
    tk.Label(master, text="Unique case ID").grid(row=15, column=0)
    tk.Label(master, text="Event names").grid(row=15, column=1)
    tk.Label(master, text="Event timestamps").grid(row=15, column=2)
    tk.Label(master, text="").grid(row=6)
    tk.Label(master, text="(Optional) Additional Features to train on").grid(row=18)
    tk.Label(master, text="Max number of epochs to train on \n(integer)").grid(row=11)
    tk.Label(master, text="File name for datasets including extension").grid(row=0, column=0)
    tk.Label(master, text="Training dataset").grid(row=1, column=0)
    tk.Label(master, text="Test dataset").grid(row=1, column=1)
    tk.Label(master, text="Output csv file name").grid(row=1, column=2)
    tk.Label(master, text="Progress of the program will be printed in the terminal").grid(row=21, column=0)
    tk.Label(master, text="Specify which epoch to load \n(default is last)").grid(row=12, column=1)
    tk.Label(master, text="Specify from which epoch to continue training \n(default is last)").grid(row=12, column=3)

    global e1, e2, e3, e6, e7, e8, e9, e10, e11, e12, e13, e14, d1, d2, d3, base_model, event_pred, time_pred, epochs
    e10.insert(0, "training.csv")
    e11.insert(0, "test.csv")
    e9.insert(0, epochs)
    e1.insert(0, "case concept:name")
    e2.insert(0, "event concept:name")
    e3.insert(0, "event time:timestamp")
    e12.insert(0, "Output.csv")

    e1.grid(row=16, column=0)
    e2.grid(row=16, column=1)
    e3.grid(row=16, column=2)
    e6.grid(row=19, column=0)
    e7.grid(row=19, column=1)
    e8.grid(row=19, column=2)
    e9.grid(row=11, column=1)
    e10.grid(row=2, column=0)
    e11.grid(row=2, column=1)
    e12.grid(row=2, column = 2)
    e13.grid(row=13, column=1)
    e14.grid(row=13, column=3)

    tk.Button(master, text='Confirm variables', command=master.quit).grid(row=20, column=0, sticky=tk.W, ipadx=5, pady=4)
    master.mainloop()

    base_features = [e1.get(), e2.get(), e3.get()]
    extra_features = [e6.get(), e7.get(), e8.get()]
    extra_features = [x for x in extra_features if x != '']

    return base_features, extra_features

base_features, extra_features = User_inputGUI()

loadEpoch = ''
trainEpoch= ''
if (loadPreviousModels.get() == 1):
    if e13.get() == '':
        loadEpoch = 0
    else:
        loadEpoch= int(e13.get())

if trainPreviousModels.get() == 1:
    if (e14.get() == ''):
        trainEpoch = 0
    else:
        trainEpoch = int(e14.get())

# Convert csv into dataframe
print("Loading datasets")
df_training_raw = pd.read_csv(dirname + "/data/" + e10.get())
df_test_raw = pd.read_csv(dirname + "/data/" + e11.get())
df_test_columns = df_test_raw.columns
# Parsing data
print("Parsing and splitting the data")
(df_training, df_2012_last_event_per_case_train) = parseData(df_training_raw)
(df_test, df_2012_last_event_per_case_test) = parseData(df_test_raw)

# Clean and split the data into train, validation & test data
(df_training, df_validation, df_test) = dataSplitter(df_training, df_test)

# Apply the naive predictors to all the datasets
print("Finding actual next event and time to next event")
(df_training, df_test) = naiveTimeToNextEventPredictor(df_training, df_test, base_features)
(df_training, df_test) = naiveNextEventPredictor(df_training, df_test, base_features)
(df_training, df_validation) = naiveTimeToNextEventPredictor(df_training, df_validation, base_features)
(df_training, df_validation) = naiveNextEventPredictor(df_training, df_validation, base_features)

if "event" in base_model.get():
    if ("LSTM" in event_pred.get()):
        print("Starting training for the LSTM model regarding events")
        accuracy, df_test = LSTMEvent(df_training, df_validation, df_test, base_features, extra_features, int(e9.get()), loadEpoch, trainEpoch)
        print('The prediction accuracy of the LSTM model for events is: {}%'.format(round(accuracy * 100, 1)))
        print('To visualize and track model\'s graph during training, how tensors over time and much more! \nrun \'tensorboard --logdir jobdir_event/logs\' in terminal.\n')
        print(u'\u2500' * 250)
    elif ("forest" in event_pred.get()):
        print("Starting training for random forest regarding events")
        accuracy, df_test = run_full_rf(df_training, df_test, base_features)
        print('The prediction accuracy of random forest for events is: {}\n%'.format(round(accuracy * 100, 1)))
        print(u'\u2500' * 250)

if "time" in base_model.get():
    if ("LSTM" in time_pred.get()):
        print("Starting training for the LSTM model regarding time")
        RMSE, df_test = LSTMTime(df_training, df_validation, df_test, base_features, extra_features, int(e9.get()), loadEpoch, trainEpoch)
        print('The RMSE of the LSTM model for time is: {} seconds'.format(round(RMSE, 1)))
        print('To visualize and track model\'s graph during training, how tensors over time and much more! \nrun \'tensorboard --logdir jobdir_time/logs\' in terminal.\n')
        print(u'\u2500' * 250)
    elif ("Multi" in time_pred.get()):
        print("Starting training for the multivariate regression model regarding time")
        RMSE, df_test, R2 = RegModel(df_training, df_test, base_features)
        print('The R2 of the multivariate regression model for time is: {} seconds\n'.format(round(R2, 1)))
        print('The RMSE of the multivariate regression model for time is: {} seconds\n'.format(round(RMSE, 1)))
        print(u'\u2500' * 250)

for x in df_test.columns:
    if (x not in df_test_columns):
        if not (x == "timePrediction" or x == "eventPrediction" or x == "naive_predicted_time_to_next_event" or x == "naive_predicted_next_event"):
            df_test.drop(columns=x, inplace=True)

print(df_test.head(10))
print("Outputting csv file to: " + (dirname + "/output/" + e12.get()))
df_test.to_csv(dirname + "/output/" + e12.get(), index=False)
print("Finished processing request!!")
master.destroy()