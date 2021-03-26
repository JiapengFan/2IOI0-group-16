import os
import re
import shutil
import tensorflow as tf
from tensorflow import keras
from .predictionAlgo import naiveNextEventPredictor, naiveTimeToNextEventPredictor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
from math import ceil
from .lstmInput import eventInputLstm, timeInputLstm
import matplotlib.pyplot as plt

def LSTMEvent(df_training_raw, df_validation_raw, df_test_raw, core_features_input: list, extra_features: list, epochs = 10):
    # Determine actual next event
    (df_training, df_validation) = naiveNextEventPredictor(df_training_raw, df_validation_raw)
    (df_training, df_test) = naiveNextEventPredictor(df_training, df_test_raw)

    unique_training_events = df_training['event concept:name'].unique().reshape(-1, 1)

    core_features = core_features_input[0:2].copy()

    core_features.append('actual_next_event')
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

    # window size as the mean of case length
    number_events_mean = df_training.groupby('case concept:name').count()['event concept:name'].mean()
    number_events_mean = ceil(number_events_mean)

    x_train, y_train = eventInputLstm(df_training, core_features, extra_features, encoder_scaler, number_events_mean)
    x_val, y_val = eventInputLstm(df_validation, core_features, extra_features, encoder_scaler, number_events_mean)
    x_test, y_test = eventInputLstm(df_test, core_features, extra_features, encoder_scaler, number_events_mean)

    CLASS_SIZE = unique_training_events.shape[0]

    def model_fn(labels_dim):
        """Create a Keras Sequential model with layers."""
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(256, input_shape=(number_events_mean, unique_training_events.shape[0])))
        model.add(keras.layers.Dropout(0.20))
        model.add(keras.layers.Dense(labels_dim, activation='softmax'))
        model.summary()
        compile_model(model)
        return model

    def compile_model(model):
        model.compile(loss='categorical_crossentropy', 
                    optimizer='adam',
                    metrics=['accuracy'])
        return model

    def get_latest(checkpoint_dir, filetype='.h5', signature='cp', overwrite=False, removeall = False):
        """ 
            This is a workaround as tf.train.latest_checkpoint does not seem to
            work well on codalab. Give preference to that function when possible.

            If overwrite is True, the latest checkpoint is reset to 0 and all 
            others are deleted.
        """
        if removeall:
            for filename in os.listdir(checkpoint_dir):
                os.remove(os.path.join(checkpoint_dir, filename))
            return None
        else:
            latest = None
            latest_number = -1
            for filename in os.listdir(checkpoint_dir):
                reference, extension = os.path.splitext(filename)
                if extension == filetype and reference.startswith('cp'):
                    number = int(re.sub(r"\D", "", reference))
                    if number > latest_number:
                        latest = filename
                        latest_number = number
                    else:
                        if overwrite:
                            os.remove(os.path.join(checkpoint_dir, filename))
            if latest is None:
                raise ValueError('No previous checkpoint found.')
            if overwrite:
                os.rename(os.path.join(checkpoint_dir, latest), os.path.join(checkpoint_dir, 'cp-0000.h5'))
                latest = 'cp-0000.h5'
                shutil.rmtree(os.path.join(checkpoint_dir, 'logs')) 
            return os.path.join(checkpoint_dir, latest)

    FILE_PATH="cp-{epoch:04d}.h5"
    LSTM_MODEL = 'lstm.h5'

    def run(num_epochs=epochs,  # Maximum number of epochs on which to train
            train_batch_size=128,  # Batch size for training steps
            job_dir='../jobdir', # Local dir to write checkpoints and export model
            checkpoint_epochs='epoch',  #  Save checkpoint every epoch
            removeall=False):
    
        """ This function trains the model for a number of epochs and returns the 
            training history. The model is periodically saved for later use.

            You can load a pre-trained model with 
                `model.load_weights(cp_path)`
            where `model` is a keras object (e.g. as returned by `model_fn`) and 
            `cp_path` is the path for the checkpoint you want to load.
            
            Setting load_previous_model to True will remove all training checkpoints.
        
        """
        
        tf.keras.backend.clear_session()

        try:
            os.makedirs(job_dir)
        except:
            pass

        checkpoint_path = FILE_PATH
        checkpoint_path = os.path.join(job_dir, checkpoint_path)

        lstm_model = model_fn(CLASS_SIZE)
        if removeall:
            latest = get_latest(job_dir, removeall=True)
            lstm_model.load_weights(latest)

        # Model checkpoint callback
        checkpoint = keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            verbose=2,
            save_freq=checkpoint_epochs,
            mode='max')

        # Tensorboard logs callback
        tblog = keras.callbacks.TensorBoard(
            log_dir=os.path.join(job_dir, 'logs'),
            histogram_freq=0,
            update_freq='epoch',
            write_graph=True,
            embeddings_freq=0)

        #     #implemented earlystopping
        #     callbacks = [checkpoint, tblog, keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6)]

        callbacks = [checkpoint, tblog]

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

        return history

    history = run(load_previous_model=False, removeall=True)

    # Plot accuracy
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Plot loss
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')

    lstm_model = model_fn(CLASS_SIZE)
    latest = get_latest('..\job_dir', overwrite=True)
    lstm_model.load_weights(latest)

    y_pred = lstm_model.predict(x_test)
    df_test['predicted_next_event_lstm'] = y_pred

    # Store to csv
    df_test.to_csv('.\predCSV\event_test_pred.csv')

    accuracy = accuracy_score[y_test, y_pred]

    return accuracy

# (df_training, df_validation) = naiveTimeToNextEventPredictor(df_training, df_validation)
# (df_training, df_test) = naiveTimeToNextEventPredictor(df_training, df_test)

# unique_training_events = df_training['event concept:name'].unique().reshape(-1, 1)

# # Define One-hot encoder for events
# onehot_encoder_event = OneHotEncoder(sparse=False, handle_unknown='ignore')
# onehot_encoder_event = onehot_encoder_event.fit(unique_training_events)

# def eventDay(dataSet):
#     dataSet["day"] = pd.to_datetime(dataSet["event time:timestamp"]).dt.day
# def eventStartHour(dataSet):
#     dataSet["hour"] = pd.to_datetime(dataSet["event time:timestamp"]).dt.hour
    
# eventDay(df_training)
# eventStartHour(df_training)

# # Normalise time to next event.
# # Hard coded column can be replaced as argument
# time_to_next_scaler = MinMaxScaler(feature_range=(0,1))
# time_to_next_event = df_training['actual_time_to_next_event'].to_numpy().reshape(-1, 1)
# time_to_next_scaler = time_to_next_scaler.fit(time_to_next_event)

# # see function comment for format
# core_features = ['case concept:name', 'event concept:name', 'actual_time_to_next_event']

# # Example with unix_reg_time as extra features
# extra_features = ['event lifecycle:transition', 'day']

# # now determining which feature of the extra features needs to be normalised becomes possible by inspecting the index of scalers list
# encoder_scaler = [onehot_encoder_event, time_to_next_scaler] + [0]*len(extra_features)

# # Instatiate scalers for features consisting out of the type float or int or encoder for categorical featurees
# for idx, extra_feature in enumerate(extra_features):
#     if len(df_training[extra_feature].unique()) < 20:
#         encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
#         arr_to_be_encoded = df_training[extra_feature].to_numpy().reshape(-1, 1)
#         encoder = encoder.fit(arr_to_be_encoded)
#         encoder_scaler[idx + 2] = encoder
#     else:
#         scaler = MinMaxScaler(feature_range=(0,1))
#         arr_to_be_normalied = df_training[extra_feature].to_numpy().reshape(-1, 1)
#         scaler = scaler.fit(arr_to_be_normalied)
#         encoder_scaler[idx + 2] = scaler
        
# # window size as the mean of case length
# number_events_mean = df_training.groupby('case concept:name').count()['event concept:name'].mean()
# number_events_mean = ceil(number_events_mean)

# x_train, y_train = timeInputLstm(df_training, core_features, extra_features, encoder_scaler, number_events_mean)