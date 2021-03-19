def lstmTimePredictor(trainDataset, testDataset):

    # Import the libraries
    import tensorflow as tf
    tf.test.gpu_device_name()

    import math
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    plt.style.use('fivethirtyeight')

    # features = (np.random.randint(10, size=(100, 1)))
    # print(features.shape)

    df_training = pd.read_csv('/content/BPI2012Training.csv')
    df_test = pd.read_csv('/content/BPI2012Test.csv')

    import datetime as dt

    # Function that parses the incoming data set

    def parseData(dataSet):
        # Parse time zone if there are any
        def convertToUnix(x):
            # If there is a timezone in the timestamp
            if 'T' in x:
                # Remove the T
                without_timezone = x[:10] + ' ' + x[11:-6]

                # Parse milliseconds if contained
                if '.' in x:
                    wholesomeTime = dt.datetime.timestamp(
                        dt.datetime.strptime(without_timezone, "%Y-%m-%d %H:%M:%S.%f"))
                else:
                    wholesomeTime = dt.datetime.timestamp(
                        dt.datetime.strptime(without_timezone, "%Y-%m-%d %H:%M:%S"))

            else:
                if '.' in x:
                    wholesomeTime = dt.datetime.timestamp(
                        dt.datetime.strptime(x, "%d-%m-%Y %H:%M:%S.%f"))
                else:
                    wholesomeTime = dt.datetime.timestamp(
                        dt.datetime.strptime(x, "%d-%m-%Y %H:%M:%S"))

            return wholesomeTime

        # Convert absolute event and reg timestamp into unix time
        dataSet['unix_abs_event_time'] = dataSet['event time:timestamp'].apply(
            lambda x: convertToUnix(x))
        dataSet['unix_reg_time'] = dataSet['case REG_DATE'].apply(
            lambda x: convertToUnix(x))

        # Time it takes for an event to occur from registeration
        dataSet['unix_rel_event_time'] = dataSet['unix_abs_event_time'] - \
            dataSet['unix_reg_time']

        # Group data set by case ID
        dataSet_grouped_by_case = dataSet.groupby(by=['case concept:name'])

        # Return data frame consisting out of the last event per case with column that indicates the number of events the case underwent appended
        dataSet_last_event_per_case = dataSet_grouped_by_case.nth([-1])
        dataSet_last_event_per_case['num_events'] = dataSet_grouped_by_case.count(
        ).iloc[:, 0]

        return (dataSet, dataSet_last_event_per_case)

    def timeToNextEvent(dataSet):
        df_predicted_time_to_next_event = dataSet.copy().sort_values(
            by=['case concept:name', "eventID ", "unix_abs_event_time"])

        unique_events = df_predicted_time_to_next_event["event concept:name"].unique(
        )

        df_new = df_predicted_time_to_next_event[[
            "actual_time_to_next_event", "unix_abs_event_time", "unix_rel_event_time", "hour", "day"]]

        for x in unique_events:
            df_new[x] = np.where(
                (df_predicted_time_to_next_event["event concept:name"] == x), 1, 0)

        return (df_new)

    def eventStartHour(dataSet):
        dataSet["hour"] = pd.to_datetime(
            dataSet["event time:timestamp"]).dt.hour

    def eventDay(dataSet):
        dataSet["day"] = pd.to_datetime(dataSet["event time:timestamp"]).dt.day

    df_training = parseData(trainingDataset)
    eventStartHour(df_training[0])
    eventDay(df_training[0])
    df_training = timeToNextEvent(df_training[0])
    df_training["finished"] = np.where(
        (np.isnan(df_training["actual_time_to_next_event"])), 1, 0)
    df_training["actual_time_to_next_event"] = np.where((np.isnan(
        df_training["actual_time_to_next_event"])), 0, df_training["actual_time_to_next_event"])

    df_test = parseData(testDataset)
    eventStartHour(df_test[0])
    eventDay(df_test[0])
    df_test = timeToNextEvent(df_test[0])
    df_test["finished"] = np.where(
        (np.isnan(df_test["actual_time_to_next_event"])), 1, 0)
    df_test["actual_time_to_next_event"] = np.where((np.isnan(
        df_test["actual_time_to_next_event"])), 0, df_test["actual_time_to_next_event"])

    f_test["W_Wijzigen contractgegevens"] = 0

    def quick(fini):
        fini["no_event"] = 0
        arrti = []
        counterquick = 0
        for x in fini:
            arrti.append(counterquick)
            if (x == 0):
                counterquick += 1
            else:
                counterquick = 0
        return arrti[:-1]

    # Convert to values using .values
    training_val = df_training.to_numpy()
    training_val = training_val.astype('float32')
    test_val = df_test.to_numpy()
    test_val = test_val.astype('float32')

    testiii = []
    for x in test_val[0:, 0]:
        testiii.append([x])

    scalerTraining = MinMaxScaler(feature_range=(0, 1))
    training_val = scalerTraining.fit_transform(training_val)
    scalerTest = MinMaxScaler(feature_range=(0, 1))
    test_val = scalerTest.fit_transform(test_val)
    scalerTestSingle = MinMaxScaler(feature_range=(0, 1))
    testSingle_val = scalerTestSingle.fit_transform(testiii)

    def create_dataset(dataset):
        dataX, dataY = [], []
        for x in range(0, len(dataset)):
            dataX.append(dataset[x: x+1, 1:])
            if (np.isnan(dataset[x][0])):
                dataY.append(0)
            else:
                dataY.append(dataset[x][0])

        return np.array(dataX), np.array(dataY)

    x_train, y_train = create_dataset(training_val)
    x_test, y_test = create_dataset(test_val)

    x_train = np.reshape(
        x_train, (x_train.shape[0], x_train.shape[2], x_train.shape[1]))
    x_test = np.reshape(
        x_test, (x_test.shape[0], x_test.shape[2], x_test.shape[1]))

    from keras.layers import Dropout
    from keras.layers.advanced_activations import LeakyReLU

    # Initialising the RNN
    model = Sequential()

    model.add(LSTM(units=x_train.shape[2], return_sequences=True, input_shape=(
        x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    # Adding a second LSTM layer and Dropout layer
    model.add(LSTM(units=16, return_sequences=True))
    model.add(Dropout(0.2))

    # Adding a third LSTM layer and Dropout layer
    model.add(LSTM(units=8, return_sequences=True))
    model.add(Dropout(0.2))

    # Adding a fourth LSTM layer and and Dropout layer
    model.add(LSTM(units=3))
    model.add(Dropout(0.2))

    # Adding the output layer
    # For Full connection layer we use dense
    # As the output is 1D so we use unit=1
    model.add(Dense(units=1))

    # model.add(LeakyReLU(alpha=0.3))

    model.summary()

    # compile and fit the model on 30 epochs
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.fit(x_train, y_train, epochs=10, batch_size=50, verbose=2)

    predictions = model.predict(x_test)
    # print(predictions[0])
    predictions = scalerTestSingle.inverse_transform(predictions)
    # print(predictions[0])
    y_test_unscaled = scalerTestSingle.inverse_transform([y_test])

    sum = 0
    for x in range(0, len(y_test)):
        sum += ((predictions[x][0] - y_test_unscaled[0][x])**2)

    print(np.sqrt(sum))

    quickarr = []
    for x in predictions:
        quickarr.append(x[0])

    testDataset['lstmTime'] = quickarr

    # print(predictions[0][0])
    # print(y_test[0][0])
    # print(predictions[1][0])
    # print(y_test[0][1])
    # print(predictions[5][0])
    # print(y_test[0][5])
    # print(predictions[100][0])
    # print(y_test[0][100])

    return testDataset
