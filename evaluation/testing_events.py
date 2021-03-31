import pandas as pd
import numpy as np

def event_accuracy(data, actualtime_event :str, prediction_event :str):
    acc = 0
    for i in range(len(data)):
        if data[actualtime_event][i] == data[prediction_event][i]:
            acc += 1
    
    accuracy = acc/len(data) 
    return accuracy

def confusion_matrix_event(actual_list, predicted_list, events: list):
    '''
        Returns df in confusion matrix format.

        Arguments:
        actual_list <type: 'np.array'>: Array with actual events, every entry can be associated with predicted_list's corresponding entry. 
        predicted_list <type: 'np.array'>: Array with predicted events. 
        events <type: list>: List of events of interest.

        Returns:
        confusion_matrix <type: 'pd.DataFrame'>: Dataframe in confusion matrix format.
    '''
    print('Im here')
    confusion_matrix = pd.DataFrame(index = events, columns = events)
    confusion_matrix.fillna(0, inplace=True)

    dataframe_elements = np.stack((actual_list, predicted_list), axis=-1)
    print('Im here')

    # Loop through actual events
    for element in dataframe_elements:
        confusion_matrix.at[element[1], element[0]] += 1

    return confusion_matrix

# Testing case 1
# actual_list = np.array(['hi1', 'hi2', 'hi3'])
# predicted_list = np.array(['hi2',  'hi3', 'hi4'])
# events = ['hi1', 'hi2',  'hi3', 'hi4']

# print(confusion_matrix_event(actual_list, predicted_list, events))
