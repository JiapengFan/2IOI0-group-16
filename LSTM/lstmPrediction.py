import os
import shutil
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.layers import Dense, LSTM
from keras.layers import Dropout
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from math import ceil
import math
from .lstmInput import eventInputLstm, timeInputLstm
import matplotlib.pyplot as plt
import warnings
import numpy as np
import datetime as dt
import pandas as pd
import keras
import re

def removesuffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    else:
        raise ValueError()

def LSTMEvent(df_training, df_validation, df_test, core_features_input: list, extra_features: list, epochs, load_epoch, train_epoch):

    warnings.filterwarnings("ignore")

    def model_fn(labels_dim, number_events_mean, unique_training_events):
        """Create a Keras Sequential model with layers."""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(128,  return_sequences=True, input_shape=(number_events_mean,
                                            unique_training_events.shape[0] + len(extra_features))))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.LSTM(64, kernel_initializer='glorot_uniform'))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(tf.keras.layers.Dense(labels_dim, activation='softmax'))
        model.summary()
        compile_model(model)
        return model

    def compile_model(model):
        model.compile(loss='categorical_crossentropy', 
                    optimizer='adam',
                    metrics=['accuracy'])
        return model

    def get_epoch(target_epoch, checkpoint_dir, filetype='.h5', signature='cp', overwrite=False):
        """ 
            If overwrite is True, the target checkpoint is reset to 0 and all 
            others are deleted.
        """

        for filename in os.listdir(checkpoint_dir):
            reference, extension = os.path.splitext(filename)
            if extension == filetype and reference.startswith('cp'):
                number = int(re.sub(r"\D", "", reference))
                if number == target_epoch:
                    target_file = filename
                else:
                    if overwrite:
                        os.remove(os.path.join(checkpoint_dir, filename))
        if target_file is None:
            raise ValueError('The given checkpoint is not found.')
        if overwrite:
            os.rename(os.path.join(checkpoint_dir, target_file), os.path.join(checkpoint_dir, 'cp-0000.h5'))
            latest = 'cp-0000.h5'
            shutil.rmtree(os.path.join(checkpoint_dir, 'logs')) 
        return os.path.join(checkpoint_dir, latest)

    file_prefix = '_'.join(extra_features) + '_'
    FILE_PATH = file_prefix + "cp-{epoch:04d}.h5"

    if load_epoch != '':
        if len(os.listdir('jobdir_event')) == 0:
            raise Exception('Please train a model before trying to load a trained model.')
        else:
            for filename in os.listdir('jobdir_event'):
                try:
                    if removesuffix(filename, 'cp-0001.h5') == file_prefix:
                        break
                    else:
                        print('Im here')
                        assert removesuffix(filename, 'cp-0001.h5') == file_prefix, 'The model that is loading is trained on different optional features, please check your optional features.'
                except ValueError:
                    pass
        
    LSTM_MODEL = 'lstm.h5'

    def run(unique_training_events,
            num_epochs=epochs,  # Maximum number of epochs on which to train
            train_batch_size=64,  # Batch size for training steps
            job_dir='jobdir_event', # Local dir to write checkpoints and export model
            checkpoint_epochs='epoch',  #  Save checkpoint every epoch
            epoch_to_train = '',
            removeall=False,
            number_events_mean = 23):
    
        """ This function trains the model for a number of epochs and returns the 
            training history. The model is periodically saved for later use.

            You can load a pre-trained model with 
                `model.load_weights(cp_path)`
            where `model` is a keras object (e.g. as returned by `model_fn`) and 
            `cp_path` is the path for the checkpoint you want to load.
            
            Setting removeall to True will remove all training checkpoints.
        
        """
        
        tf.keras.backend.clear_session()

        try:
            os.makedirs(job_dir)
        except:
            pass

        CLASS_SIZE = unique_training_events.shape[0]
    
        checkpoint_path = FILE_PATH
        checkpoint_path = os.path.join(job_dir, checkpoint_path)

        lstm_model = model_fn(CLASS_SIZE, number_events_mean, unique_training_events)
        if epoch_to_train == 0:
            path = os.path.join(job_dir, 'lstm.h5')
            lstm_model.load_weights(path)

        if removeall or epoch_to_train == 0:
            for filename in os.listdir(job_dir):
                try:
                    os.remove(os.path.join(job_dir, filename))
                except:
                    shutil.rmtree(os.path.join(job_dir, filename))
        else:
            target_epoch = get_epoch(epoch_to_train, job_dir, overwrite=True)
            lstm_model.load_weights(target_epoch)

        # Model checkpoint callback
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            verbose=2,
            save_freq=checkpoint_epochs,
            mode='max')

        # Tensorboard logs callback
        tblog = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(job_dir, 'logs'),
            histogram_freq=0,
            update_freq='epoch',
            write_graph=True,
            embeddings_freq=0)

        #     #implemented earlystopping
        #     callbacks = [checkpoint, tblog, keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6)]

        callbacks = [checkpoint, tblog, tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4)]

        history = lstm_model.fit(
                x=x_train,
                y=y_train, 
                validation_data = (x_val, y_val),
                batch_size=train_batch_size,
                steps_per_epoch=None,
                epochs=num_epochs,
                callbacks=callbacks,
                verbose=2)
        
        lstm_model.save(os.path.join(job_dir, LSTM_MODEL))

        return lstm_model, history

    current_unique = df_training['event concept:name'].unique()
    next_unique = df_training['actual_next_event'].unique()
    unique_training_events = np.append(next_unique, np.setdiff1d(current_unique, next_unique, assume_unique=True)).reshape(-1, 1)

    core_features = core_features_input[0:2].copy()

    core_features.append('actual_next_event')

    print('Instantiating scalers and one-hot encoders for features...')
    # Define One-hot encoder for events
    onehot_encoder_event = OneHotEncoder(sparse=False, handle_unknown='ignore')
    onehot_encoder_event = onehot_encoder_event.fit(unique_training_events)

    # now determining which feature of the extra features needs to be normalised becomes possible by inspecting the index of scalers list
    encoder_scaler = [onehot_encoder_event] + [0]*len(extra_features)

    # Instatiate scalers for features consisting out of the type float or int or encoder for categorical featurees
    for idx, extra_feature in enumerate(extra_features):
        if len(df_training[extra_feature].unique()) < 20:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            arr_to_be_encoded = df_training[extra_feature].to_numpy().reshape(-1, 1)
            encoder = encoder.fit(arr_to_be_encoded)
            encoder_scaler[idx + 1] = encoder
        else:
            scaler = MinMaxScaler(feature_range=(0,1))
            arr_to_be_normalied = df_training[extra_feature].to_numpy().reshape(-1, 1)
            scaler = scaler.fit(arr_to_be_normalied)
            encoder_scaler[idx + 1] = scaler

    print('Done instantiating scalers and one-hot encoders for features!')

    # window size as the mean of case length
    number_events_mean = df_training.groupby('case concept:name').count()['event concept:name'].mean()
    number_events_mean = ceil(number_events_mean)

    print('Converting input to lstm accepted format...')
    if (load_epoch == ''):
        x_train, y_train = eventInputLstm(df_training, core_features, extra_features, encoder_scaler, number_events_mean)
        x_val, y_val = eventInputLstm(df_validation, core_features, extra_features, encoder_scaler, number_events_mean)
    x_test, y_test = eventInputLstm(df_test, core_features, extra_features, encoder_scaler, number_events_mean)
    print('Done converting input to lstm accepted format!')

    CLASS_SIZE = unique_training_events.shape[0]

    if load_epoch == '' and train_epoch == '':
        print('Training LSTM model...')
        lstm_model, _ = run(unique_training_events, removeall=True, number_events_mean=number_events_mean)
        print('Done training LSTM model!')
    else:
        if load_epoch == 0:
            print('loading trained LSTM model...')
            lstm_model = model_fn(CLASS_SIZE, number_events_mean, unique_training_events)
            lstm_model.load_weights('./jobdir_event/lstm.h5')
            print('Done loading trained LSTM model!')
        else:
            print('loading trained LSTM model...')
            lstm_model = model_fn(CLASS_SIZE, number_events_mean, unique_training_events)
            lstm_model.load_weights(f'./jobdir_event/cp-{load_epoch:04d}.h5')
            print('Done loading trained LSTM model!')
                
        if train_epoch != '':
            print('loading trained LSTM model and starts training...')
            lstm_model, _ = run(unique_training_events, epoch_to_train=train_epoch, number_events_mean=number_events_mean)
            print('Done loading trained LSTM model and finished training!')

    y_pred_class = lstm_model.predict_classes(x_test, batch_size=64)
    y_pred_ohe = np.array([[0] * len(unique_training_events) for _ in y_pred_class])
    for idx, class_pred in enumerate(y_pred_class):
        y_pred_ohe[idx, class_pred] = 1
    
    y_pred = onehot_encoder_event.inverse_transform(y_pred_ohe)
    y_pred = np.ravel(y_pred)
    df_test['eventPrediction'] = y_pred.tolist()

    accuracy = accuracy_score(y_test, y_pred_ohe)
    return accuracy, df_test


def LSTMTime(dataset, validationDataset, applyDataset, core_features, extra_features, epochs, load_epoch, train_epoch):

    warnings.filterwarnings("ignore")

    def model_fn(number_events_mean, unique_training_events):
        """Create a Keras Sequential model with layers."""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(256, input_shape=(number_events_mean,
                                            unique_training_events.shape[0] + len(extra_features)), return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.20))
        model.add(tf.keras.layers.LSTM(units=1, activation='linear'))
        model.summary()
        compile_model(model)
        return model

    def compile_model(model):
        model.compile(loss='mse', 
                    optimizer='adam',
                    metrics=[tf.keras.metrics.RootMeanSquaredError()])
        return model

    def get_epoch(target_epoch, checkpoint_dir, filetype='.h5', signature='cp', overwrite=False):
        """ 
            If overwrite is True, the target checkpoint is reset to 0 and all 
            others are deleted.
        """

        for filename in os.listdir(checkpoint_dir):
            reference, extension = os.path.splitext(filename)
            if extension == filetype and reference.startswith('cp'):
                number = int(re.sub(r"\D", "", reference))
                if number == target_epoch:
                    target_file = filename
                else:
                    if overwrite:
                        os.remove(os.path.join(checkpoint_dir, filename))
        if target_file is None:
            raise ValueError('The given checkpoint is not found.')
        if overwrite:
            os.rename(os.path.join(checkpoint_dir, target_file), os.path.join(checkpoint_dir, 'cp-0000.h5'))
            latest = 'cp-0000.h5'
            shutil.rmtree(os.path.join(checkpoint_dir, 'logs')) 
        return os.path.join(checkpoint_dir, latest)

    file_prefix = '_'.join(extra_features) + '_'
    FILE_PATH = file_prefix + "cp-{epoch:04d}.h5"

    if load_epoch != '' or train_epoch != '':
        if len(os.listdir('jobdir_time')) == 0:
            raise Exception('Please train a model before trying to load a trained model.')
        else:
            for filename in os.listdir('jobdir_time'):
                try:
                    if removesuffix(filename, 'cp-0001.h5') == file_prefix:
                        break
                    else:
                        assert removesuffix(filename, 'cp-0001.h5') == file_prefix, 'The model that is loading is trained on different optional features, please check your optional features.'
                except Exception:
                    pass
        
    LSTM_MODEL = 'lstm.h5'

    def run(unique_training_events,
            num_epochs=epochs,  # Maximum number of epochs on which to train
            train_batch_size=64,  # Batch size for training steps
            job_dir='jobdir_time', # Local dir to write checkpoints and export model
            checkpoint_epochs='epoch',  #  Save checkpoint every epoch
            epoch_to_train = '',
            removeall=False,
            number_events_mean = 23):
    
        """ This function trains the model for a number of epochs and returns the 
            training history. The model is periodically saved for later use.

            You can load a pre-trained model with 
                `model.load_weights(cp_path)`
            where `model` is a keras object (e.g. as returned by `model_fn`) and 
            `cp_path` is the path for the checkpoint you want to load.
            
            Setting removeall to True will remove all training checkpoints.
        
        """
        
        tf.keras.backend.clear_session()

        try:
            os.makedirs(job_dir)
        except:
            pass

        checkpoint_path = FILE_PATH
        checkpoint_path = os.path.join(job_dir, checkpoint_path)

        lstm_model = model_fn(number_events_mean, unique_training_events)
        if epoch_to_train == 0:
            lstm_model.load_weights('./jobdir_event/lstm.h5')

        if removeall or epoch_to_train == 0:
            for filename in os.listdir(job_dir):
                try:
                    os.remove(os.path.join(job_dir, filename))
                except:
                    shutil.rmtree(os.path.join(job_dir, filename))
        else:
            target_epoch = get_epoch(epoch_to_train, job_dir, overwrite=True)
            lstm_model.load_weights(target_epoch)

        # Model checkpoint callback
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            verbose=2,
            save_freq=checkpoint_epochs,
            mode='max')

        # Tensorboard logs callback
        tblog = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(job_dir, 'logs'),
            histogram_freq=0,
            update_freq='epoch',
            write_graph=True,
            embeddings_freq=0)

    #implemented earlystopping
        callbacks = [checkpoint, tblog, tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', patience=4)]

        history = lstm_model.fit(
                x=x_train,
                y=y_train, 
                validation_data = (x_val, y_val),
                batch_size=train_batch_size,
                steps_per_epoch=None,
                epochs=num_epochs,
                callbacks=callbacks,
                verbose=2)
        
        lstm_model.save(os.path.join(job_dir, LSTM_MODEL))

        return lstm_model, history
    
    # Convert csv into dataframe
    df_training = dataset.copy()
    df_validation = validationDataset.copy()
    df_test = applyDataset.copy()

    def eventTimeConverter(inputData):
        inputData["day"] = pd.to_datetime(inputData[core_features[2]]).dt.day % 7
        inputData["hour"] = pd.to_datetime(inputData[core_features[2]]).dt.hour
        return inputData

    print("Adding hour of day and day of week for training...")
    df_training = eventTimeConverter(df_training)
    df_validation = eventTimeConverter(df_validation)
    df_test = eventTimeConverter(df_test)
    print("Done adding hour of day and day of week for training!")

    current_unique = df_training[core_features[1]].unique()
    next_unique = df_training['actual_next_event'].unique()
    unique_training_events = np.append(next_unique, np.setdiff1d(current_unique, next_unique, assume_unique=True)).reshape(-1, 1)

    # Define One-hot encoder for events
    print('Instantiating scalers and one-hot encoders for features...')
    onehot_encoder_event = OneHotEncoder(sparse=False, handle_unknown='ignore')
    onehot_encoder_event = onehot_encoder_event.fit(unique_training_events)

    # Normalise time to next event.
    # Hard coded column can be replaced as argument
    time_to_next_scaler = MinMaxScaler(feature_range=(0,1))
    time_to_next_event = df_training['actual_time_to_next_event'].to_numpy().reshape(-1, 1)
    time_to_next_scaler = time_to_next_scaler.fit(time_to_next_event)

    # see function comment for format
    core_features = core_features[0:2]
    core_features.append("actual_time_to_next_event")

    # now determining which feature of the extra features needs to be normalised becomes possible by inspecting the index of scalers list
    encoder_scaler = [onehot_encoder_event, time_to_next_scaler] + [0]*len(extra_features)

    # Instatiate scalers for features consisting out of the type float or int or encoder for categorical featurees
    for idx, extra_feature in enumerate(extra_features):
        if len(df_training[extra_feature].unique()) < 25:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            arr_to_be_encoded = df_training[extra_feature].to_numpy().reshape(-1, 1)
            encoder = encoder.fit(arr_to_be_encoded)
            encoder_scaler[idx + 2] = encoder
        else:
            scaler = MinMaxScaler(feature_range=(0,1))
            arr_to_be_normalied = df_training[extra_feature].to_numpy().reshape(-1, 1)
            scaler = scaler.fit(arr_to_be_normalied)
            encoder_scaler[idx + 2] = scaler
    print('Done instantiating scalers and one-hot encoders for features!')

    # window size as the mean of case length
    number_events_mean = df_training.groupby('case concept:name').count()['event concept:name'].mean()
    number_events_mean = ceil(number_events_mean)

    print('Converting input to lstm accepted format...')
    if (load_epoch == ''):
        x_train, y_train = timeInputLstm(df_training, core_features, extra_features, encoder_scaler, number_events_mean)
        x_val, y_val = timeInputLstm(df_validation, core_features, extra_features, encoder_scaler, number_events_mean)
    x_test, y_test = timeInputLstm(df_test, core_features, extra_features, encoder_scaler, number_events_mean)
    print('Done converting input to lstm accepted format!')

    if load_epoch == '' and train_epoch == '':
        print('Training LSTM model...')
        lstm_model, _ = run(unique_training_events, removeall=True, number_events_mean=number_events_mean)
        print('Done training LSTM model!')
    else:
        if load_epoch == 0:
            print('loading trained LSTM model...')
            lstm_model = model_fn()
            lstm_model.load_weights('./jobdir_event/lstm.h5')
            print('Done loading trained LSTM model!')
        else:
            print('loading trained LSTM model...')
            lstm_model = model_fn()
            lstm_model.load_weights(f'./jobdir_event/cp-{load_epoch:04d}.h5')
            print('Done loading trained LSTM model!')
        
        if train_epoch != '':
            print('loading trained LSTM model and starts training...')
            lstm_model, _ = run(unique_training_events, epoch_to_train=train_epoch, number_events_mean=number_events_mean)
            print('Done loading trained LSTM model and finished training!')


    predictions = lstm_model.predict(x_test)
    predictions_unscaled = time_to_next_scaler.inverse_transform(predictions)
    y_test_unscaled = time_to_next_scaler.inverse_transform(y_test)

    RMSE = math.sqrt(mean_squared_error(
        y_test_unscaled[0:, 0], predictions_unscaled[0:, 0]))

    applyDataset["timePrediction"] = predictions_unscaled

    return RMSE, applyDataset