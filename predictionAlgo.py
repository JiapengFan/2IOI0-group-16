import pandas as pd
from dataParsing import parseData

# Convert csv into dataframe
df_2012 = pd.read_csv('.\data\BPI2012Training.csv')
df_2012_test = pd.read_csv('.\data\BPI2012Test.csv')

# Parse data
(df_2012, df_2012_last_event_per_case) = parseData(df_2012)
(df_2012_test, df_2012_last_event_per_case_Test) = parseData(df_2012_test)


# Function that naively predicts the avereage time that each case takes in days
def naiveAverageTimeOfCasePredictor(dataSet_last_event_per_case):

    # Create a safe working copy of the input dataset
    df_prediction_temp = dataSet_last_event_per_case.copy()

    # Convert relative time in seconds to relative time in days
    prediction_time_of_case = df_prediction_temp['unix_rel_event_time'].mean(
    ) / 3600 / 24

    return prediction_time_of_case


# Function that naively predicts the time till the next event
def naiveTimeToNextEventPredictor(dataSet, applyDataSet):

    # Create a safe working copy of the input dataset
    df_prediction_temp = dataSet.copy()

    # Dictionary holding the best prediction of the next event given the current event
    bestPredicitonForEvent = {}

    # Dictionary holding the amount of occurrences oe ach event after a given event
    event_dictionary = {}

    # Array holding the actual next events of each event in a case, if it is not the last event
    actualNextEvent = []

    # Series with all the unique events that are present in the dataset
    unique_events = df_prediction_temp['event concept:name'].unique()

    # Add all the keys to the event_dictionary and the bestPredictionForEvent dictionary
    for key in unique_events:
        event_dictionary[key] = [0, 0]
        bestPredicitonForEvent[key] = 0

    # Sort the temporary datafrime to use time analysis for each dase
    df_predicted_next_event = df_prediction_temp.sort_values(
        by=['case concept:name']).reset_index(drop=True)

    # Get the amount of occurrences of each event after a specific event and store this information in the event_dictionary
    for index, row in df_predicted_next_event.iterrows():
        caseID = row['case concept:name']
        try:
            nextRow = df_predicted_next_event.iloc[index+1]
        except:
            continue
        if (caseID != nextRow['case concept:name']):
            continue
        else:
            #print(event_dictionary[row['event concept:name']])
            event_dictionary[row['event concept:name']][0] += (
                nextRow['unix_abs_event_time'] - row['unix_abs_event_time'])
            event_dictionary[row['event concept:name']][1] += 1

    for key in event_dictionary:
        bestPredicitonForEvent[key] = event_dictionary[key][0] / \
            event_dictionary[key][1]

    print(bestPredicitonForEvent)

    applyDataSet['timeToNextEvent'] = applyDataSet['event concept:name'].map(
        bestPredicitonForEvent)

    return applyDataSet


# Function to naively predict the next event
def naiveNextEventPredictor(dataSet, applyDataSet):

    # Create a safe working copy of the input dataset
    df_prediction_temp = dataSet.copy()

    # Dictionary holding the best prediction of the next event given the current event
    bestPredicitonForEvent = {}

    # Dictionary holding the amount of occurrences oe ach event after a given event
    event_dictionary = {}

    # Array holding the actual next events of each event in a case, if it is not the last event
    actualNextEvent = []

    # Series with all the unique events that are present in the dataset
    unique_events = df_prediction_temp['event concept:name'].unique()

    # Add all the keys to the event_dictionary and the bestPredictionForEvent dictionary
    for key in unique_events:
        event_dictionary[key] = {}
        bestPredicitonForEvent[key] = ""
        for key2 in unique_events:
            event_dictionary[key][key2] = 0

    # Sort the temporary datafrime to use time analysis for each dase
    df_predicted_next_event = df_prediction_temp.sort_values(
        by=['case concept:name']).reset_index(drop=True)

    # Get the amount of occurrences of each event after a specific event and store this information in the event_dictionary
    for index, row in df_predicted_next_event.iterrows():
        caseID = row['case concept:name']
        try:
            nextRow = df_predicted_next_event.iloc[index+1]
        except:
            continue
        else:
            event_dictionary[row['event concept:name']
                             ][nextRow['event concept:name']] += 1

    # Sort the temporary datafrime holding the testing data to use time analysis for each dase
    df_actual_next_event_apply_dataset = applyDataSet.sort_values(
        by=['case concept:name']).reset_index(drop=True)

    # Get the actual next events for the testing data
    for index, row in df_actual_next_event_apply_dataset.iterrows():
        caseID = row['case concept:name']
        try:
            nextRow = df_actual_next_event_apply_dataset.iloc[index+1]
        except:
            actualNextEvent.append("nan")
            continue
        if (caseID != nextRow['case concept:name']):
            actualNextEvent.append("nan")
            continue
        else:
            actualNextEvent.append(nextRow['event concept:name'])

    # Calculate which event is the most likely to occur after each specific event
    for key in event_dictionary:
        currentMax = 0
        for key2 in event_dictionary[key]:
            if event_dictionary[key][key2] > currentMax:
                currentMax = event_dictionary[key][key2]
                bestPredicitonForEvent[key] = key2

    # Add the predicted next evet to the applyDataset dataframe
    applyDataSet['predictedNextEvent'] = applyDataSet['event concept:name'].map(
        bestPredicitonForEvent)

    # Add the actual next event to the dataframe
    applyDataSet['actualNextEvent'] = actualNextEvent

    return(applyDataSet)


print(naiveTimeToNextEventPredictor(df_2012, df_2012_test).head(20))
#print(naiveNextEventPredictor(df_2012, df_2012_test).head(20))
# print(naiveAverageTimeOfCasePredictor(df_2012_last_event_per_case))
