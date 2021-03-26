from keras.preprocessing.sequence import pad_sequences
import numpy as np

def eventInputLstm(df, core_features: list, extra_features: list, scaler_encoder: list, window_size: int):
    '''
    args:
    df <class: 'pd.DataFrame'>: Dataframe of interest, filled with the features of arguments.
    core_features <class: 'list'>: Column names of the core features in format of [case id, current event, next event].
    extra_features <class: 'list'>: Extra features that the user wishes to train on, can be given in any sequences.
    scaler_encoder <class: 'list' of 'sklearn.preprocessing.MinMaxScaler' or klearn.preprocessing.OneHotEncoder>: 
    Trained scaler or encoder associated with the input features by their index that is used to normalise float or int features and encode categorical features.
    window_size <class: 'int'>: Window size of lstm input.
    
    returns:
    x_arr <'np.array'>: LSTM input in format [samples, timestep, features].
    y_arr <'np.array'>: LSTM input in format [samples, features].
    '''
    case_id_col = core_features[0]
    event_id_col = core_features[1]
    y_output_col = core_features[2]
    
    scaler_encoder_copy = scaler_encoder.copy()
    
    encoder_events = scaler_encoder_copy.pop(0)
    
    # Prevent modifying argument
    relevant_columns = core_features.copy()
    try:
        relevant_columns.extend(extra_features)
    except:
        print('Please input valid features.')
    
    df_relevant = df[relevant_columns].copy()

    # One-hot encode current and next event
    current_event = df_relevant[event_id_col].to_numpy().reshape(-1, 1)
    df_relevant[event_id_col] = encoder_events.transform(current_event).tolist()

    next_event = df_relevant[y_output_col].to_numpy().reshape(-1, 1)
    df_relevant[y_output_col] = encoder_events.transform(next_event).tolist()
    
    # Normalise extra features that are ints or floats
    for idx, scaler in enumerate(scaler_encoder_copy):
        if scaler != 0:
            to_be_normalised = df_relevant[extra_features[idx]].to_numpy().reshape(-1, 1)
            df_relevant[extra_features[idx]] = scaler.transform(to_be_normalised)

    # Prepare input and output in form of [samples, features]
    x = []
    y = []

    # Get groupby object df by case id
    df_groupby_case_id = df_relevant.groupby(case_id_col)

    # Unique case ids
    unique_case_ids = df_relevant[case_id_col].unique().tolist()

    # Find input and output vector in form of [samples, features]
    for unique_id in unique_case_ids:
        columns_relevant = [event_id_col, y_output_col].copy()
        columns_relevant.extend(extra_features)
        xy_unique_id = df_groupby_case_id.get_group(unique_id)[columns_relevant].values.tolist()
        sequence_x_unique_id = []

        # event[0] = current event, event[1] = next event, event[2] = first selected feature, event[3] = second selected feature,...
        for idx, event in enumerate(xy_unique_id):
            if len(sequence_x_unique_id) == window_size:
                del sequence_x_unique_id[0]

            event_memory = event[0].copy()
            event_memory.extend(event[2:])
            sequence_x_unique_id.append(event_memory.copy())
            x.append(sequence_x_unique_id.copy())
            y.append(event[1])
    
    # Alter input to [samples, timestep, features] for lstm, zero padding used to equalize timestep length
    x_arr = pad_sequences(x, dtype='float32')
    
    # Convert y to format [samples, features]
    y_arr = np.reshape(y, (-1, len(y[0])))

    return x_arr, y_arr

def timeInputLstm(df, core_features: list, extra_features: list, scaler_encoder: list, window_size: int):
    '''
    args:
    df <class: 'pd.DataFrame'>: Dataframe of interest, filled with the features of arguments.
    core_features <class: 'list'>: Column names of the core features in format of [case id, current event, time to next event].
    extra_features <class: 'list'>: Extra features that the user wishes to train on, can be given in any sequences.
    scaler_encoder <class: 'list' of 'sklearn.preprocessing.MinMaxScaler' or klearn.preprocessing.OneHotEncoder>: 
    Trained scaler or encoder associated with the input features by their index that is used to normalise float or int features and encode categorical features.
    window_size <class: 'int'>: Window size of lstm input.
    
    returns:
    x_arr <'np.array'>: LSTM input in format [samples, timestep, features].
    y_arr <'np.array'>: LSTM input in format [samples, features].
    '''
    # See function comment for format
    case_id_col = core_features[0]
    event_id_col = core_features[1]
    y_output_col = core_features[2]
    
    scaler_encoder_copy = scaler_encoder.copy()
    
    encoder_events = scaler_encoder_copy.pop(0)

    time_to_next_scaler = scaler_encoder_copy.pop(0)
    
    # Prevent modifying argument
    relevant_columns = core_features.copy()
    relevant_columns.extend(extra_features)
    
    try:
        df_relevant = df[relevant_columns].copy()
    except:
        print('Please input valid features.')

    # One-hot encode current event
    current_event = df_relevant[event_id_col].to_numpy().reshape(-1, 1)
    df_relevant[event_id_col] = encoder_events.transform(current_event).tolist()

    # Normalise time to next event.
    time_to_next_event = df_relevant[y_output_col].to_numpy().reshape(-1, 1)
    df_relevant[y_output_col] = time_to_next_scaler.transform(time_to_next_event)
    
    for idx, scaler_encoder in enumerate(scaler_encoder_copy):
            to_be_normalised_or_encoded = df_relevant[extra_features[idx]].to_numpy().reshape(-1, 1)
            df_relevant[extra_features[idx]] = scaler_encoder.transform(to_be_normalised_or_encoded)

    # Prepare input and output in form of [samples, features]
    x = []
    y = []

    # Get groupby object df by case id
    df_groupby_case_id = df_relevant.groupby(case_id_col)

    # Unique case ids
    unique_case_ids = df_relevant[case_id_col].unique().tolist()
    
    # Remove case id out of relevant columns
    del relevant_columns[0]

    # Find input and output vector in form of [samples, features]
    for unique_id in unique_case_ids:
        xy_unique_id = df_groupby_case_id.get_group(unique_id)[relevant_columns].values.tolist()
        sequence_x_unique_id = []

        # event[0] = current event, event[1] = actual time to next event, event[2] = first selected feature, event[3] = second selected feature,...
        for event in xy_unique_id:
            if len(sequence_x_unique_id) == window_size:
                del sequence_x_unique_id[0]

            event_memory = event[0].copy()
            event_memory.extend(event[2:])
            sequence_x_unique_id.append(event_memory.copy())
            x.append(sequence_x_unique_id.copy())
            y.append(event[1])
    
    # Alter input to [samples, timestep, features] for lstm, zero padding used to equalize timestep length
    x_arr = pad_sequences(x, dtype='float32')
    
    # Convert y to format [samples, features]
    y_arr = np.reshape(y, (-1, 1))

    return x_arr, y_arr