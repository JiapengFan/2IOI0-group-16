import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.visualization.petrinet import visualizer as pn_visualizer

# Convert csv into dataframe
df_2012 = pd.read_csv('..\.\data\BPI2012Training.csv').rename(columns={'case concept:name': 'case:concept:name', 'event concept:name': 'concept:name', 'event time:timestamp' : 'time:timestamp'})
df_2012_Test = pd.read_csv('..\.\data\BPI2012Test.csv')

# Parse data
# (df_2012, df_2012_last_event_per_case) = parseData(df_2012)
# (df_2012_Test, df_2012_last_event_per_case_Test) = parseData(df_2012_Test)

df_2012 = dataframe_utils.convert_timestamp_columns_in_df(df_2012)
df_2012 = df_2012.sort_values(by='time:timestamp')

log = log_converter.apply(df_2012)

# alpha miner
net, initial_marking, final_marking = alpha_miner.apply(log)

# add information about frequency to the viz 
parameters = {pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png"}
gviz = pn_visualizer.apply(net, initial_marking, final_marking, 
                           parameters=parameters, 
                           variant=pn_visualizer.Variants.FREQUENCY, 
                           log=log)

# save the Petri net
pn_visualizer.save(gviz, "alpha_miner_petri_net.png")