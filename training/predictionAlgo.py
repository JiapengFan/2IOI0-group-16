import numpy as np

# Function that naively predicts the time till the next event
def naiveTimeToNextEventPredictor(dataSet, applyDataSet, coreFeatures):

    df_predicted_time_to_next_event = dataSet.copy().sort_values(
        by=[coreFeatures[0], "unix_abs_event_time"])

    applyDataSet.sort_values(
        by=[coreFeatures[0], "unix_abs_event_time"], inplace=True)

    df_predicted_time_to_next_event['actual_time_to_next_event'] = df_predicted_time_to_next_event['unix_abs_event_time'].shift(
        -1) - df_predicted_time_to_next_event["unix_abs_event_time"]

    applyDataSet["actual_time_to_next_event"] = applyDataSet["unix_abs_event_time"].shift(
        -1) - applyDataSet["unix_abs_event_time"]

    df_predicted_time_to_next_event['actual_time_to_next_event'] = np.where((df_predicted_time_to_next_event[coreFeatures[0]] == df_predicted_time_to_next_event[coreFeatures[0]].shift(
        -1)), df_predicted_time_to_next_event['actual_time_to_next_event'], 0)

    applyDataSet['actual_time_to_next_event'] = np.where((applyDataSet[coreFeatures[0]] == applyDataSet[coreFeatures[0]].shift(
        -1)), applyDataSet['actual_time_to_next_event'], 0)
    # print(df_predicted_next_event.index.get_level_values(0))

    df_prediction_time_temp = df_predicted_time_to_next_event.copy()
    df_prediction_time_temp_grouped = df_prediction_time_temp.groupby(
        by=coreFeatures[1]).mean()["actual_time_to_next_event"]

    # rint(df_prediction_time_temp_grouped.head(50))

    df_prediction_time_temp['naive_predicted_time_to_next_event'] = df_prediction_time_temp[coreFeatures[1]].map(
        df_prediction_time_temp_grouped)

    applyDataSet['naive_predicted_time_to_next_event'] = applyDataSet[coreFeatures[1]].map(
        df_prediction_time_temp_grouped)

    dataSet = df_prediction_time_temp

    return dataSet, applyDataSet

# Function to naively predict the next event


def naiveNextEventPredictor(dataSet, applyDataSet, coreFeatures):

    df_predicted_next_event = dataSet.copy().sort_values(
        by=[coreFeatures[0], "unix_abs_event_time"])

    applyDataSet.sort_values(
        by=[coreFeatures[0], "unix_abs_event_time"], inplace=True)

    df_predicted_next_event['actual_next_event'] = df_predicted_next_event[coreFeatures[1]].shift(
        -1)

    applyDataSet["actual_next_event"] = applyDataSet[coreFeatures[1]].shift(
        -1)

    df_predicted_next_event['actual_next_event'] = np.where((df_predicted_next_event[coreFeatures[0]] == df_predicted_next_event[coreFeatures[0]].shift(
        -1)), df_predicted_next_event['actual_next_event'], np.nan)

    applyDataSet['actual_next_event'] = np.where((applyDataSet[coreFeatures[0]] == applyDataSet[coreFeatures[0]].shift(
        -1)), applyDataSet['actual_next_event'], np.nan)

    df_event_temp = df_predicted_next_event.copy()
    unique_events = df_event_temp[coreFeatures[1]].unique()
    for x in unique_events:
        df_event_temp[x] = np.where(
            (df_event_temp["actual_next_event"] == x), 1, 0)

    df_event_temp_grouped = df_event_temp.groupby(
        by=coreFeatures[1]).sum()

    df_event_temp_grouped = df_event_temp_grouped[unique_events]

    df_predicted_next_event['naive_predicted_next_event'] = df_predicted_next_event[coreFeatures[1]].map(
        df_event_temp_grouped.idxmax(axis=1))

    applyDataSet['naive_predicted_next_event'] = applyDataSet[coreFeatures[1]].map(
        df_event_temp_grouped.idxmax(axis=1))

    dataSet = df_predicted_next_event

    return dataSet, applyDataSet
