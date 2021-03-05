from memory_profiler import profile # import this package
from training.predictionAlgo import naiveNextEventPredictor, naiveTimeToNextEventPredictor, naiveAverageTimeOfCasePredictor
from preprocessing.dataParsing import parseData
import pandas as pd
import timeit

@profile(precision=4) #place profile before the function, this will return memory use when running the function
def function():
    # Convert csv into dataframe
    df_2012 = pd.read_csv('.\data\BPI2012Training.csv')
    df_2012_Test = pd.read_csv('.\data\BPI2012Test.csv')

    # Parse data
    (df_2012, df_2012_last_event_per_case) = parseData(df_2012)
    (df_2012_Test, df_2012_last_event_per_case_Test) = parseData(df_2012_Test)

    naiveNextEventPredictor(df_2012, df_2012_Test)
    naiveTimeToNextEventPredictor(df_2012, df_2012_Test)
    naiveAverageTimeOfCasePredictor(df_2012_last_event_per_case)

starttime = timeit.default_timer()
print("The start time is :",starttime)
function()
print("The time difference is :", timeit.default_timer() - starttime)