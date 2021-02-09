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

print(PredictionTimeDays)
print(PredictionTimeHours, '\n')
print(PredictionTimeDays, '\n')

# DF that removed first and last event of each case
df_2012_removed_first_last_event = df_2012[~df_2012.index.isin(df_2012.groupby(by=['case concept:name']).nth([0, -1]).index)]

hi = df_2012.groupby(by=['case concept:name']).nth([2])

hi1 = hi['event concept:name'].value_counts()

print(hi['event concept:name'].unique())
