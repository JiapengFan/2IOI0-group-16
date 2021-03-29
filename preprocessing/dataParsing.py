import datetime as dt
import pandas as pd
import numpy as np

# Function that parses the incoming data set


def parseData(dataSet):
    # Parse time zone if there are any
    def convertToUnix(x):
        # If there is a timezone in the timestamp
        if 'T' in x:
            # Remove the T
            without_timezone = x[:10] + ' ' + x[11:-6]

            # Parse milliseconds if contained
            if '.' in x:
                wholesomeTime = dt.datetime.timestamp(
                    dt.datetime.strptime(without_timezone, "%Y-%m-%d %H:%M:%S.%f"))
            else:
                wholesomeTime = dt.datetime.timestamp(
                    dt.datetime.strptime(without_timezone, "%Y-%m-%d %H:%M:%S"))

        else:
            if '.' in x:
                wholesomeTime = dt.datetime.timestamp(
                    dt.datetime.strptime(x, "%d-%m-%Y %H:%M:%S.%f"))
            else:
                wholesomeTime = dt.datetime.timestamp(
                    dt.datetime.strptime(x, "%d-%m-%Y %H:%M:%S"))

        return wholesomeTime

    # Convert absolute event and reg timestamp into unix time
    dataSet['unix_abs_event_time'] = dataSet['event time:timestamp'].apply(
        lambda x: convertToUnix(x))

    # Time it takes for an event to occur from registeration

    dataSet.sort_values(["case concept:name", "unix_abs_event_time"], inplace=True)
    tempi = dataSet.groupby(by="case concept:name").first()["unix_abs_event_time"]
    dataSet["unix_reg_time"] = dataSet["case concept:name"].map(tempi)
    dataSet["unix_rel_event_time"] = dataSet["unix_abs_event_time"] - dataSet["case concept:name"].map(tempi)

    # Group data set by case ID
    dataSet_grouped_by_case = dataSet.groupby(by=['case concept:name'])

    # Return data frame consisting out of the last event per case with column that indicates the number of events the case underwent appended
    dataSet_last_event_per_case = dataSet_grouped_by_case.nth([-1])
    dataSet_last_event_per_case['num_events'] = dataSet_grouped_by_case.count(
    ).iloc[:, 0]

    return (dataSet, dataSet_last_event_per_case)