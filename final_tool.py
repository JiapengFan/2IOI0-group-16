
import pandas as pd
import numpy as np
from preprocessing.dataParsing import parseData
from preprocessing.dataSplitting import dataSplitter
import tkinter as tk
from tkinter import *
from LSTM.lstmPrediction import LSTMEvent
import warnings

warnings.filterwarnings("ignore")

def User_inputGUI():
    master = tk.Tk()
    tk.Label(master, text="possible feature synonyms:").grid(row=10)
    tk.Label(master, text="case ID").grid(row=11, column=0)
    tk.Label(master, text="event name").grid(row=11, column=1)
    tk.Label(master, text="event time").grid(row=11, column=2)
    tk.Label(master, text="case reg date").grid(row=11, column=3)
    tk.Label(master, text="Required amount").grid(row=11, column=4)
    tk.Label(master, text="").grid(row=3)
    tk.Label(master, text="Possible extra features:").grid(row=14)
    tk.Label(master, text="LSTM epochs (integer)").grid(row=8)

    e9 = tk.Entry(master)
    e9.grid(row=8, column=1)

    e1 = tk.Entry(master)
    e2 = tk.Entry(master)
    e3 = tk.Entry(master)
    e4 = tk.Entry(master)
    e5 = tk.Entry(master)

    e1.grid(row=12, column=0)
    e2.grid(row=12, column=1)
    e3.grid(row=12, column=2)
    e4.grid(row=12, column=3)
    e5.grid(row=12, column=4)


    tk.Label(master, text='Time prediction').grid(row=4, column=0)
    time_pred = StringVar(master)
    time_pred.set("LSTM")

    tk.Label(master, text='Event prediction').grid(row=4, column=2)
    event_pred = StringVar(master)
    event_pred.set("LSTM")

    d2 = OptionMenu(master, time_pred, 'LSTM', 'naive predictor').grid(
        row=4, column=1)

    d3 = OptionMenu(master, event_pred, 'LSTM', 'Random forest', 'naive predictor').grid(
        row=4, column=3)

    tk.Label(master, text="").grid(row=9)
    tk.Label(master, text='model selection').grid(row=0)
    base_model = StringVar(master)
    base_model.set("Time and event prediction")

    d1 = OptionMenu(master, base_model, 'event_prediction', 'time prediction', 'time and event prediction').grid(
        row=0, column=1)

    #w.pack()

    tk.Button(master, text='Confirm variables', command=master.quit).grid(row=16, column=0, sticky=tk.W, pady=4)


    tk.Label(master, text="").grid(row=3)
    tk.Label(master, text="Possible extra features:").grid(row=14)

    e6 = tk.Entry(master)
    e7 = tk.Entry(master)
    e8 = tk.Entry(master)

    e6.grid(row=15, column=0)
    e7.grid(row=15, column=1)
    e8.grid(row=15, column=2)

    master.mainloop()

    f1, f2, f3, f4, f5 = e1.get(), e2.get(), e3.get(), e4.get(), e5.get()

    if f1 == "":
        f1 = 'case concept:name'

    if f2 == "":
       f2 = 'event concept:name'

    if f3 == "":
        f3 = 'event time:timestamp'

    if f4 == "":
        f4 = 'case REG_DATE'

    if f5 == "":
        f5 = 'case AMOUNT_REQ'

    epochs = e9.get()
    if epochs == '':
        epochs = 20
    else:
        epochs = int(epochs)

    base_features = [f1, f2, f3, f4, f5]

    extra_features = [e6.get(), e7.get(), e8.get()]
    extra_features = [x for x in extra_features if x != '']

    selected_model = [base_model.get()]
    model_types = [time_pred.get(), event_pred.get()]

    print(base_features)


    return base_features, extra_features, epochs, selected_model, model_types

base_features, extra_features, epochs, selected_model, predictors = User_inputGUI()

# Convert csv into dataframe
df_training_raw = pd.read_csv('.\data\BPI2012Training.csv')
df_test_raw = pd.read_csv('.\data\BPI2012Test.csv')

# Parsing data
(df_training, df_2012_last_event_per_case_train) = parseData(df_training_raw)
(df_test, df_2012_last_event_per_case_test) = parseData(df_test_raw)

# Clean and split the data into train, validation & test data
(df_training, df_validation, df_test) = dataSplitter(df_training, df_test)

if selected_model == 'event prediction':
    accuracy, df_test = LSTMEvent(df_training, df_validation, df_test, base_features, extra_features)

    print('The prediction accuracy of LSTM for events is: {}'.format(accuracy))

    print(df_test)

    print('To visualize and track model\'s graph during training, how tensors over time and much more! \nrun \'tensorboard --logdir jobdir_event/logs\' in terminal.')
# core_features = ['case concept:name', 'event concept:name', 'event time:timestamp', 'case REG_DATE']

# # Example extra features
# extra_features = ['event lifecycle:transition', 'case AMOUNT_REQ']

## Load tensorboard
# tensorboard --logdir jobdir/logs