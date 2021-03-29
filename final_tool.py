
import pandas as pd
import numpy as np
from preprocessing.dataParsing import parseData
from preprocessing.dataSplitting import dataSplitter
import tkinter as tk
from tkinter import *
from LSTM.lstmPrediction import LSTMEvent

def User_inputGUI():
    master = tk.Tk()
    tk.Label(master, text="Core features:").grid(row=0)
    tk.Label(master, text="eventID").grid(row=1, column=0)
    tk.Label(master, text="event name").grid(row=1, column=1)
    tk.Label(master, text="event time").grid(row=1, column=2)
    tk.Label(master, text="case reg date").grid(row=1, column=3)
    tk.Label(master, text="").grid(row=3)
    tk.Label(master, text="Optional extra features:").grid(row=4)

    e1 = tk.Entry(master)
    e2 = tk.Entry(master)
    e3 = tk.Entry(master)
    e4 = tk.Entry(master)
    e5 = tk.Entry(master)
    e6 = tk.Entry(master)
    e7 = tk.Entry(master)

    e1.grid(row=2, column=0)
    e2.grid(row=2, column=1)
    e3.grid(row=2, column=2)
    e4.grid(row=2, column=3)
    e5.grid(row=5, column=0)
    e6.grid(row=5, column=1)
    e7.grid(row=5, column=2)

    tk.Label(master, text='model selection').grid(row=7)
    base_model = StringVar(master)
    base_model.set("time and event prediction")

    w = OptionMenu(master, base_model, 'event prediction', 'time predictiom', 'time and event prediction').grid(
        row=7, column=1)
    #w.pack()

    tk.Button(master, text='Confirm variables', command=master.quit).grid(row=9, column=0, sticky=tk.W, pady=4)

    master.mainloop()

    f1, f2, f3, f4 = e1.get(), e2.get(), e3.get(), e4.get()

    if f1 == "":
        f1 = 'case concept:name'

    if f2 == "":
        f2 = 'event concept:name'

    if f3 == "":
        f3 = 'event time:timestamp'

    if f4 == "":
        f4 = 'case REG_DATE'

    base_features = [f1, f2, f3, f4]

    extra_features = [e5.get(), e6.get(), e7.get()]
    extra_features = [x for x in extra_features if x != '']

    selected_model = base_model.get()

    return base_features, extra_features, selected_model

base_features, extra_features, selected_model,  = User_inputGUI()

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

# core_features = ['case concept:name', 'event concept:name', 'event time:timestamp', 'case REG_DATE']

# # Example with unix_reg_time as extra features
# extra_features = ['event lifecycle:transition', 'case AMOUNT_REQ']