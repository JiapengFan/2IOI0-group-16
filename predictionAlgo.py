import pandas as pd
from dataParsing import parseData 

# Convert csv into dataframe
df_2012 = pd.read_csv('.\data\BPI2012Test.csv')
df_Italy = pd.read_csv('.\data\BPI2012Test.csv')

# Parse data
(df_2012, df_2012_last_event_per_case)= parseData(df_2012)

#PredictionTime in sec
PredictionTime = df_2012_last_event_per_case['unix_rel_event_time'].mean()

#PredictionTime in hours
PredictionTimeHours = PredictionTime / 3600

PredictionTimeDays = PredictionTimeHours / 24

print(PredictionTimeHours, '\n')
print(PredictionTimeDays, '\n')