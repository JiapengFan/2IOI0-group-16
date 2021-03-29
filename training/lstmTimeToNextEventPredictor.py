# Import the libraries
import tensorflow as tf
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
import datetime as dt
import keras
from sklearn.metrics import mean_squared_error
import math
from math import ceil


def lstmTimePredictor(dataset, validationDataset, applyDataset, coreFeatures, extraFeatures):

    # Convert csv into dataframe
    df_training = dataset.copy()
    df_validation = validationDataset.copy()
    df_test = applyDataset.copy()

    def eventTimeConverter(inputData):
        inputData["day"] = pd.to_datetime(inputData[coreFeatures[2]]).dt.day % 7
        inputData["hour"] = pd.to_datetime(inputData[coreFeatures[2]]).dt.hour
        return inputData

    df_training = eventTimeConverter(df_training)
    df_validation = eventTimeConverter(df_validation)
    df_test = eventTimeConverter(df_test)

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

        
        # Prevent modifying argument
        relevant_columns = core_features.copy()
        relevant_columns.extend(extra_features)
        
        try:
            df_relevant = df[relevant_columns].copy()
        except:
            print('Please input valid features.')


        case_id_col = core_features[0]
        event_id_col = core_features[1]
        y_output_col = core_features[2]
        
        scaler_encoder_copy = scaler_encoder.copy()
        
        encoder_events = scaler_encoder_copy.pop(0)

        time_to_next_scaler = scaler_encoder_copy.pop(0)

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
                if len(sequence_x_unique_id) == number_events_mean:
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

    unique_training_events = df_training['event concept:name'].unique().reshape(-1, 1)

    # Define One-hot encoder for events
    onehot_encoder_event = OneHotEncoder(sparse=False, handle_unknown='ignore')
    onehot_encoder_event = onehot_encoder_event.fit(unique_training_events)

    # Normalise time to next event.
    # Hard coded column can be replaced as argument
    time_to_next_scaler = MinMaxScaler(feature_range=(0,1))
    time_to_next_event = df_training['actual_time_to_next_event'].to_numpy().reshape(-1, 1)
    time_to_next_scaler = time_to_next_scaler.fit(time_to_next_event)

    # see function comment for format
    core_features = coreFeatures[0:2]
    core_features.append("actual_time_to_next_event")

    # now determining which feature of the extra features needs to be normalised becomes possible by inspecting the index of scalers list
    encoder_scaler = [onehot_encoder_event, time_to_next_scaler] + [0]*len(extraFeatures)

    # Instatiate scalers for features consisting out of the type float or int or encoder for categorical featurees
    for idx, extra_feature in enumerate(extraFeatures):
        if len(df_training[extra_feature].unique()) < 20:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            arr_to_be_encoded = df_training[extra_feature].to_numpy().reshape(-1, 1)
            encoder = encoder.fit(arr_to_be_encoded)
            encoder_scaler[idx + 2] = encoder
        else:
            scaler = MinMaxScaler(feature_range=(0,1))
            arr_to_be_normalied = df_training[extra_feature].to_numpy().reshape(-1, 1)
            scaler = scaler.fit(arr_to_be_normalied)
            encoder_scaler[idx + 2] = scaler
            
    # window size as the mean of case length
    number_events_mean = df_training.groupby('case concept:name').count()['event concept:name'].mean()
    number_events_mean = ceil(number_events_mean)

    x_train, y_train = timeInputLstm(df_training, core_features, extraFeatures, encoder_scaler, number_events_mean)
    x_val, y_val = timeInputLstm(df_validation, core_features, extraFeatures, encoder_scaler, number_events_mean)
    x_test, y_test = timeInputLstm(df_test, core_features, extraFeatures, encoder_scaler, number_events_mean)

    print(np.isnan(np.sum(y_train)))
    model = Sequential()
    model.add(LSTM(256, input_shape=(number_events_mean,
                                     unique_training_events.shape[0] + len(extraFeatures)), return_sequences=True))
    model.add(keras.layers.Dropout(0.20))

    model.add(keras.layers.LSTM(units=1, activation='linear'))

    model.compile(optimizer="adam", loss='mse')
    history = model.fit(x_train, y_train, epochs=10, batch_size=256,
                        validation_data=(x_val, y_val), verbose=2, shuffle=False)

    predictions = model.predict(x_test)
    predictions_unscaled = time_to_next_scaler.inverse_transform(predictions)
    y_test_unscaled = time_to_next_scaler.inverse_transform(y_test)

    RMSE = math.sqrt(mean_squared_error(
        y_test_unscaled[0:, 0], predictions_unscaled[0:, 0]))

    applyDataset["timePrediction"] = predictions_unscaled

    return RMSE, applyDataset