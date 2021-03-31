import pandas as pd
import numpy as np
# pd.set_option('display.max_columns', None)


def timePrediction(df_training, df_validation, df_test):

    df_train = df_training.copy()
    df_train = df_train.set_index(["case concept:name", "eventID "])

    bigDict = {}
    counter = 0
    for index, df_temp in df_train.groupby(level=0):
        if (counter < -1):
            break
        previousEvents = []
        caseTime = df_temp['unix_rel_event_time'].values[-1]
        for eventID, row in df_temp.iterrows():
            previousEvents.append(row['event concept:name'])
            previousEventsTuple = str(previousEvents)
            if previousEventsTuple in bigDict:
                if (np.isnan(row["actual_time_to_next_event"])):
                    bigDict[previousEventsTuple].append(int(0))
                else:
                    bigDict[previousEventsTuple].append(
                        caseTime - row["unix_rel_event_time"])
            else:
                if (np.isnan(row["actual_time_to_next_event"])):
                    bigDict[previousEventsTuple] = [int(0)]
                else:
                    bigDict[previousEventsTuple] = [
                        caseTime - row["unix_rel_event_time"]]
        counter += 1

    # print(list(bigDict.items())[:2])

    df_test = df_test.set_index(["case concept:name", "eventID "])
    df_test.sort_values(["unix_abs_event_time"])
    df_test['research'] = 0

    for index, df_temp in df_test.groupby(level=0):
        previousEvents = []
        eventid = 0
        lastPrediction = -1

        eventCounter = 1
        df_temp_rows = df_temp.shape[0]
        df_temp_iterrator = df_temp.iterrows()
        for eventID, row in df_temp_iterrator:
            previousEvents.append(row['event concept:name'])
            previousEventsTuple = str(previousEvents)
            if (previousEventsTuple in bigDict):
                if (str(previousEvents.append(row["actual_next_event"])) in bigDict):
                    meanCurrentEvent = sum(
                        bigDict[previousEventsTuple]) / len(bigDict[previousEventsTuple])
                    meanNextEvent = sum(str(previousEvents.append(
                        row["actual_next_event"]))) / len(str(previousEvents.append(row["actual_next_event"])))
                    lastPrediction = meanCurrentEvent - meanNextEvent
                else:
                    meanCurrentEvent = sum(
                        bigDict[previousEventsTuple]) / len(bigDict[previousEventsTuple])

                    noNextEvents = df_temp_rows - eventCounter
                    lastPrediction = meanCurrentEvent / noNextEvents
                eventid = eventID
            df_test.loc[(slice(None), eventid[1]), "research"] = lastPrediction
            eventCounter += 1

    return df_training, df_validation, df_test
