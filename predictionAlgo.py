import pandas as pd
from dataParsing import parseData

# Convert csv into dataframe
df_2012 = pd.read_csv('.\data\BPI2012Test.csv')
df_Italy = pd.read_csv('.\data\BPI2012Test.csv')

# Parse data
(df_2012, df_2012_last_event_per_case) = parseData(df_2012)

# Function that naively predicts the time till the next event


def naiveTimeToNextEventPredictor(dataSet_last_event_per_case):

    # PredictionTime in sec
    PredictionTime = dataSet_last_event_per_case['unix_rel_event_time'].mean()

    # PredictionTime in hours
    PredictionTimeHours = PredictionTime / 3600

    # PredictionTime in days
    PredictionTimeDays = PredictionTimeHours / 24

    return PredictionTimeDays


# Function to naively predict the next event
def naiveNextEventPredictor(dataSet):

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
            actualNextEvent.append("nan")
            continue
        if (caseID != nextRow['case concept:name']):
            actualNextEvent.append("nan")
            continue
        else:
            event_dictionary[row['event concept:name']
                             ][nextRow['event concept:name']] += 1
            actualNextEvent.append(nextRow['event concept:name'])

    # Calculate which event is the most likely to occur after each specific event
    for key in event_dictionary:
        currentMax = 0
        for key2 in event_dictionary[key]:
            if event_dictionary[key][key2] > currentMax:
                currentMax = event_dictionary[key][key2]
                bestPredicitonForEvent[key] = key2

    # Add the new predicted next event to the dataframe
    df_predicted_next_event['predictedNextEvent'] = df_predicted_next_event['event concept:name'].map(
        bestPredicitonForEvent)

    # Add the actual next event to the dataframe
    df_predicted_next_event['actualNextEvent'] = actualNextEvent

    return(df_predicted_next_event)


print(naiveTimeToNextEventPredictor(df_2012_last_event_per_case))
print(naiveNextEventPredictor(df_2012).head(20))
