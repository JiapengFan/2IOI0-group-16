import tkinter as tk
from tkinter import *

def User_inputGUI():
    master = tk.Tk()
    tk.Label(master, text="possible feature synonyms:").grid(row=0)
    tk.Label(master, text="eventID").grid(row=1, column=0)
    tk.Label(master, text="event name").grid(row=1, column=1)
    tk.Label(master, text="event time").grid(row=1, column=2)
    tk.Label(master, text="case reg date").grid(row=1, column=3)
    tk.Label(master, text="").grid(row=3)
    tk.Label(master, text="Possible extra features:").grid(row=4)

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

    w = OptionMenu(master, base_model, 'event_prediction', 'time predictiom', 'time and event prediction').grid(
        row=7, column=1)
    #w.pack()

    tk.Button(master, text='Confirm variables', command=master.quit).grid(row=9, column=0, sticky=tk.W, pady=4)

    master.mainloop()

    f1, f2, f3, f4 = e1.get(), e2.get(), e3.get(), e4.get()

    if f1 == "":
        f1 = 'eventID'

    if f2 == "":
        f2 = 'event concept:name'

    if f3 == "":
        f3 = 'event time:timestamp'

    if f4 == "":
        f4 = 'case REG_DATE'

    base_features = [f1, f2, f3, f4]

    extra_features = [e5.get(), e6.get(), e7.get()]
    extra_features = [x for x in extra_features if x != '']

    selected_model = [base_model.get()]


    return base_features, extra_features, selected_model

User_inputGUI()