import pandas as pd
import datetime as dt

df_2012 = pd.read_csv('.\data\BPI2012Test.csv')
df_Italy = pd.read_csv('.\data\BPI2012Test.csv')


unix_time = df_2012['event time:timestamp'].apply(lambda x: dt.datetime.timestamp(dt.datetime.strptime(x, "%d-%m-%Y %H:%M:%S.%f")))

# date = dt.datetime.strptime(date, "%d-%m-%Y %H:%M:%S.%f")

# date = dt.datetime.timestamp(date)

df_2012['unix_timestamp'] = unix_time
