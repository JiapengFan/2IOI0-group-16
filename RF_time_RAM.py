import pandas as pd
import numpy as np
from training.predictionAlgo import naiveNextEventPredictor
from preprocessing.dataParsing import parseData
from preprocessing.dataSplitting import dataSplitter
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from memory_profiler import profile
# import timeit

@profile(precision=4) #place profile before the function, this will return memory use when running the function
def function():
    # Convert csv into dataframe
    df_training_raw = pd.read_csv('.\data\BPI2012Training.csv')
    df_test_raw = pd.read_csv('.\data\BPI2012Test.csv')

    # Parsing data
    (df_training, df_2012_last_event_per_case_train) = parseData(df_training_raw)
    (df_test, df_2012_last_event_per_case_test) = parseData(df_test_raw)

    # Clean and split the data into train, validation & test data
    (df_training, df_validation, df_test) = dataSplitter(df_training, df_test)

    unique_training_events = df_training['event concept:name'].unique().reshape(-1, 1)

    # Define One-hot encoder
    onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    onehot_encoder = onehot_encoder.fit(unique_training_events)

    def createInputRF(df):
        # df with only relevant training data, i.e. loan amount, current event, next event and time elapsed since registeration.
        df_relevant = df[['case concept:name', 'event concept:name', 'actual_next_event', 'case AMOUNT_REQ', 'unix_reg_time']].copy()

        # One-hot encode current and next event
        training_current_event = df_relevant['event concept:name'].to_numpy().reshape(-1, 1)
        df_relevant['event concept:name'] = onehot_encoder.transform(training_current_event).tolist()

        training_next_event = df_relevant['actual_next_event'].to_numpy().reshape(-1, 1)
        df_relevant['actual_next_event'] = onehot_encoder.transform(training_next_event).tolist()

        # Normalise loan amount
        loan_scaler = MinMaxScaler(feature_range=(0,1))
        case_amount = df_relevant['case AMOUNT_REQ'].to_numpy().reshape(-1, 1)
        df_relevant['case AMOUNT_REQ'] = np.around(loan_scaler.fit_transform(case_amount), decimals = 4)

        # Normalise time in seconds from case registeration to current event
        time_scaler = MinMaxScaler(feature_range=(0,1))
        reg_time = df_relevant['unix_reg_time'].to_numpy().reshape(-1, 1)
        df_relevant['unix_reg_time'] = np.around(loan_scaler.fit_transform(reg_time), decimals = 4)

        # Prepare input and output in form of [samples, features]
        x = []
        y = []

        # Get groupby object df by case id
        df_groupby_case_id = df_relevant.groupby('case concept:name')

        # Unique case ids
        unique_case_ids = df_relevant['case concept:name'].unique().tolist()

        # Find input and output vector in form of [samples, features]
        for unique_id in unique_case_ids:
            xy_unique_id = df_groupby_case_id.get_group(unique_id)[['event concept:name', 'actual_next_event', 'case AMOUNT_REQ', 'unix_reg_time']].values.tolist()

            base_case = xy_unique_id[0][0:2].copy()
            x_first_sample_per_case = base_case[0].copy()
            x_first_sample_per_case.extend([xy_unique_id[0][2], xy_unique_id[0][3]])
            x.append(x_first_sample_per_case)
            y.append(base_case[1].copy())

            # event[0] = current event, event[1] = next event, event[2] = loan amount, event[3] = time elapsed since registeration of case
            for event in xy_unique_id[1:]:
                base_case[0] = [prev_xs + current_x for prev_xs, current_x in zip(base_case[0], event[0])]
                x_sample = base_case[0].copy()
                x_sample.extend([event[2], event[3]])
                x.append(x_sample)
                y.append(event[1])
        return x, y
    
    # Determine actual next event
    (df_training, df_validation) = naiveNextEventPredictor(df_training, df_validation)
    (df_test, df_validation) = naiveNextEventPredictor(df_test, df_validation)

    x_train, y_train = createInputRF(df_training)
    x_test, y_test = createInputRF(df_test)

    rf = RandomForestClassifier(n_estimators = 400, min_samples_split = 0.02, min_samples_leaf = 0.02, max_depth = 200, max_features = 'sqrt', bootstrap = True)
    rf.fit(x_train, y_train)

function()
# starttime = timeit.default_timer()
# print("The start time is :",starttime)
# function()
# print("The time difference is :", timeit.default_timer() - starttime)
