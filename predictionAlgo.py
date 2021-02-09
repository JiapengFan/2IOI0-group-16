import pandas as pd
from dataParsing import parseData 

# Convert csv into dataframe
data_2012 = pd.read_csv('.\data\BPI2012Training.csv')
data_Italy = pd.read_csv('.\data\BPI2012Training.csv')

# Parse data
[df_2012, df_2012_last_event_per_case]= parseData(data_2012)

# PredictionTime in sec
PredictionTime = df_2012_last_event_per_case['unix_rel_event_time'].mean()

# PredictionTime in hours
PredictionTimeHours = PredictionTime / 3600

PredictionTimeDays = PredictionTimeHours / 24

# hi = df_2012[~df_2012.index.isin(df_2012.groupby(by=['case concept:name']).nth(0).index)]

# hi1 = hi['event concept:name'].value_counts()
# print(PredictionTimeDays)
print(data_Italy.columns)

# print(PredictionTimeHours, '\n')
# print(PredictionTimeDays, '\n')