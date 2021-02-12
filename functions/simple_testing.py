import pandas as pd

def MSEcalc(data, actualtime_col :str, prediction_col :str):
    total_mse = 0
    
    for i in range(len(data) + 1):
        mse = ((data[prediction_col][str(i)] - data[actualtime_col][str(i)])^2)
        total_mse += mse
    
    return total_mse / len(data)
    
def confusion_matrix_time(data, actualtime_col :str, prediction_col :str, error_margin):
    correct = 0
    one_stage_above = 0
    two_plus_stages_above = 0
    one_stage_below = 0
    two_plus_stages_below = 0
    
    for i in range(len(data)):
        error = (data[prediction_col][i] - data[actualtime_col][i])
        
        if error < error_margin and error > (-1*error_margin) :
            correct += 1
            
        elif error > error_margin and error < 2*error_margin:
            one_stage_above += 1
            
        elif error > 2*error_margin:
            two_plus_stages_above += 1
            
        elif error < (1*error_margin) and error > (-2*error_margin):
            one_stage_below += 1
            
        elif error < (-2*error_margin):
            two_plus_stages_below += 1
            
    df = {'2 stages lower': [two_plus_stages_below], 'one stage below': [one_stage_below], 'correct': [correct],
            'one stage above': [one_stage_above], 'two stages above': [two_plus_stages_above]}
    df = pd.DataFrame(df)
    
    return df
        

def event_accuracy(data, actualtime_event :str, prediction_event :str):
    acc = 0
    for i in range(len(data)):
        if data[actualtime_event][i] == data[prediction_event][i]:
            acc += 1

    accuracy = acc/len(data)
    return accuracy

def confusion_matrix_event(data, actual_col :str, prediction_col :str, events):
    df = {}

    for actual_event in events:
        predicted_list = [0]*len(events)
        for row in range(len(data)):
            for idx, predicted_event in enumerate(events):
                if (data[prediction_col][row] == predicted_event) and (data[actual_col][row] == actual_event):
                    predicted_list[idx]+=1
        df[actual_event] = predicted_list
    df = pd.DataFrame(df)

    indices = {}

    for x in range(0, len(events)):
        indices[x] = events[x]

    df.rename(index = indices, inplace=True)

    return df