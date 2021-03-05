import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.util import dataframe_utils
## Import heuristics miner algorithm
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
# Import the heuristics net visualisation object
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer

#df_2012 = pd.read_csv(r'C:\Users\20190971\Desktop\ProcessMining\export_training_df.csv')
df_2012 = pd.read_csv('.\data\export_training.csv')
df_2012 = df_2012.sort_values(by=['case concept:name', 'event time:timestamp'])

df_2012 = df_2012.reset_index(drop=True)
case = df_2012.loc[0, 'case concept:name']

counter = 0
maxi = len(df_2012.index) - 1

#filtering data to only inclide cases of legth atmost 10
for i in df_2012.index:
    if df_2012.loc[i, 'case concept:name'] == case:
        counter = counter + 1
    elif df_2012.loc[i, 'case concept:name'] != case or i == maxi:
        if counter > 10:
            booleanMask = df_2012['case concept:name'] == case
            df_2012 = df_2012.drop(df_2012[booleanMask].index)
            case = df_2012.loc[i, 'case concept:name']
            counter = 1
        else :           
            counter = 1
            case = df_2012.loc[i, 'case concept:name']


df_2012.rename(columns={'event time:timestamp': 'time:timestamp', 
 'case concept:name': 'case:concept:name', 'event concept:name': 'concept:name'}, inplace=True)

testDf = df_2012[df_2012.columns[[1,5,7]]]
log = log_converter.apply(testDf)

# heuristics miner
heu_net = heuristics_miner.apply_heu(log)
# Visualise model
gviz2 = hn_visualizer.apply(heu_net)
hn_visualizer.view(gviz2)
