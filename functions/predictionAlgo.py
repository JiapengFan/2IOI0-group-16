import numpy as np

# Function that naively predicts the avereage time that each case takes in days


def naiveAverageTimeOfCasePredictor(dataSet_last_event_per_case_train, dataSet_last_event_per_case_test):

    # Create a safe working copy of the input dataset
    df_prediction_temp = dataSet_last_event_per_case_train.copy()

    # Convert relative time in seconds to relative time in days
    prediction_time_of_case = df_prediction_temp['unix_rel_event_time'].mean(
    ) / 3600 / 24

    dataSet_last_event_per_case_test['predictedTime'] = prediction_time_of_case

    return dataSet_last_event_per_case_test


# Function that naively predicts the time till the next event
def naiveTimeToNextEventPredictor(dataSet, applyDataSet):

    df_predicted_time_to_next_event = dataSet.copy().sort_values(
        by=['case concept:name', "eventID ", "unix_abs_event_time"])

    df_predicted_time_to_next_event['actual_time_to_next_event'] = df_predicted_time_to_next_event['unix_abs_event_time'].shift(
        -1) - df_predicted_time_to_next_event["unix_abs_event_time"]

    df_predicted_time_to_next_event['actual_time_to_next_event'] = np.where((df_predicted_time_to_next_event['case concept:name'] == df_predicted_time_to_next_event['case concept:name'].shift(
        -1)), df_predicted_time_to_next_event['actual_time_to_next_event'], np.nan)
    # print(df_predicted_next_event.index.get_level_values(0))

    df_prediction_time_temp = df_predicted_time_to_next_event.copy()
    df_prediction_time_temp_grouped = df_prediction_time_temp.groupby(
        by="event concept:name").mean()["actual_time_to_next_event"]

    df_prediction_time_temp['naive_predicted_time_to_next_event'] = df_prediction_time_temp['event concept:name'].map(
        df_prediction_time_temp_grouped)

    applyDataSet['naive_predicted_time_to_next_event'] = applyDataSet['event concept:name'].map(
        df_prediction_time_temp_grouped)

    dataSet = df_prediction_time_temp

    return dataSet, applyDataSet

# Function to naively predict the next event


def naiveNextEventPredictor(dataSet, applyDataSet):

    df_predicted_next_event = dataSet.copy().sort_values(
        by=['case concept:name', "eventID ", "unix_abs_event_time"])

    df_predicted_next_event['actual_next_event'] = df_predicted_next_event["event concept:name"].shift(
        -1)

    applyDataSet["actual_next_event"] = applyDataSet["event concept:name"].shift(
        -1)

    df_predicted_next_event['actual_next_event'] = np.where((df_predicted_next_event['case concept:name'] == df_predicted_next_event['case concept:name'].shift(
        -1)), df_predicted_next_event['actual_next_event'], np.nan)

    applyDataSet['actual_next_event'] = np.where((applyDataSet['case concept:name'] == applyDataSet['case concept:name'].shift(
        -1)), applyDataSet['actual_next_event'], np.nan)

    df_event_temp = df_predicted_next_event.copy()
    unique_events = df_event_temp["event concept:name"].unique()
    for x in unique_events:
        df_event_temp[x] = np.where(
            (df_event_temp["actual_next_event"] == x), 1, 0)

    df_event_temp_grouped = df_event_temp.groupby(
        by="event concept:name").sum()

    df_event_temp_grouped = df_event_temp_grouped[unique_events]

    df_predicted_next_event['naive_predicted_next_event'] = df_predicted_next_event['event concept:name'].map(
        df_event_temp_grouped.idxmax(axis=1))

    applyDataSet['naive_predicted_next_event'] = applyDataSet['event concept:name'].map(
        df_event_temp_grouped.idxmax(axis=1))

    dataSet = df_predicted_next_event

    return dataSet, applyDataSet
