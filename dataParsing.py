import pandas as pd
import numpy as np
import datetime as dt

# Parse csv into dataframe
df_2012 = pd.read_csv('.\data\BPI2012Test.csv')
df_Italy = pd.read_csv('.\data\BPI2012Test.csv')

# Parse time zone if there are any
def convertToUnix(x):
    # If there is a timezone in the timestamp
    if 'T' in x: 
        # Remove the T
        without_timezone = x[:10] + ' ' + x[11:-6]

        # Parse milliseconds if contained
        if '.' in x:
            without_timezone_unix = dt.datetime.timestamp(dt.datetime.strptime(without_timezone, "%Y-%m-%d %H:%M:%S.%f"))
        else:
            without_timezone_unix = dt.datetime.timestamp(dt.datetime.strptime(without_timezone, "%Y-%m-%d %H:%M:%S"))

        # Add timezone or remove
        if (x[-6]=='+'):
            wholesomeTime = without_timezone_unix + int(x[-5:-3]) * 3600 + int(x[-2:]) * 60
        else:
            wholesomeTime = without_timezone_unix - int(x[-5:-3]) * 3600 - int(x[-2:]) * 60
        
    else:
        if '.' in x:
            wholesomeTime = dt.datetime.timestamp(dt.datetime.strptime(x, "%d-%m-%Y %H:%M:%S.%f"))
        else:
            wholesomeTime = dt.datetime.timestamp(dt.datetime.strptime(x, "%d-%m-%Y %H:%M:%S"))

    return wholesomeTime

# Convert event and reg timestamp into unix time
df_2012['unix_event_time'] = df_2012['event time:timestamp'].apply(lambda x: convertToUnix(x))
df_2012['unix_reg_time'] = df_2012['case REG_DATE'].apply(lambda x: convertToUnix(x))

# Group data set by case ID 
df_2012_grouped_by_case = df_2012.groupby(by=['case concept:name'])

# Return data frame consisting out of the last event per case with column that indicates the number of events the case underwent appended
df_2012_last_event_per_case = df_2012_grouped_by_case.nth([-1])
df_2012_last_event_per_case['num_events'] = df_2012_grouped_object.count()['unix_event_time']

# Inspect data frames
print(df_2012_last_event_per_case, '\n')
print(df_2012)