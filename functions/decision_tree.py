import pandas as pd
import numpy as np



def dummy_variables(df):
    df_dummy = df

    for event_type in df['event concept:name'].unique()[1:]:
        df_dummy[event_type] = 0
        for event in df_dummy.index:
            if df_dummy['event concept:name'][event] == event_type:
                df_dummy[event_type][event] = 1

    return df_dummy

def dummy_trainers(dummy_data):
    x1 = dummy_data['A_PARTLYSUBMITTED'][:-1]
    x2 = dummy_data['A_PREACCEPTED'][:-1]
    x3 = dummy_data['W_Completeren aanvraag'][:-1]
    x4 = dummy_data['A_DECLINED'][:-1]
    x5 = dummy_data['W_Afhandelen leads'][:-1]
    x6 = dummy_data['A_ACCEPTED'][:-1]
    x7 = dummy_data['O_SELECTED'][:-1]
    x8 = dummy_data['A_FINALIZED'][:-1]
    x9 = dummy_data['O_CREATED'][:-1]
    x10 = dummy_data['O_SENT'][:-1]
    x11 = dummy_data['W_Nabellen offertes'][:-1]
    x12 = dummy_data['O_CANCELLED'][:-1]
    x13 = dummy_data['A_CANCELLED'][:-1]
    x14 = dummy_data['W_Beoordelen fraude'][:-1]
    x15 = dummy_data['O_SENT_BACK'][:-1]
    x16 = dummy_data['W_Valideren aanvraag'][:-1]
    x17 = dummy_data['W_Nabellen incomplete dossiers'][:-1]
    x18 = dummy_data['O_ACCEPTED'][:-1]
    x19 = dummy_data['A_APPROVED'][:-1]
    x20 = dummy_data['A_ACTIVATED'][:-1]
    x21 = dummy_data['A_REGISTERED'][:-1]
    x22 = dummy_data['O_DECLINED'][:-1]
    x23 = dummy_data['W_Wijzigen contractgegevens'][:-1]
    x_time = dummy_data['unix_rel_event_time'][1:]
    y_train = dummy_data['event concept:name'][1:]

    zipped = zip(x_time, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21,
                 x22, x23)

    X_train =[list(a) for a in zipped]

    return X_train, y_train

def x_prediction(test_data):
    x1 = test_data['A_PARTLYSUBMITTED'][:-1]
    x2 = test_data['A_PREACCEPTED'][:-1]
    x3 = test_data['W_Completeren aanvraag'][:-1]
    x4 = test_data['A_DECLINED'][:-1]
    x5 = test_data['W_Afhandelen leads'][:-1]
    x6 = test_data['A_ACCEPTED'][:-1]
    x7 = test_data['O_SELECTED'][:-1]
    x8 = test_data['A_FINALIZED'][:-1]
    x9 = test_data['O_CREATED'][:-1]
    x10 = test_data['O_SENT'][:-1]
    x11 = test_data['W_Nabellen offertes'][:-1]
    x12 = test_data['O_CANCELLED'][:-1]
    x13 = test_data['A_CANCELLED'][:-1]
    x14 = test_data['W_Beoordelen fraude'][:-1]
    x15 = test_data['O_SENT_BACK'][:-1]
    x16 = test_data['W_Valideren aanvraag'][:-1]
    x17 = test_data['W_Nabellen incomplete dossiers'][:-1]
    x18 = test_data['O_ACCEPTED'][:-1]
    x19 = test_data['A_APPROVED'][:-1]
    x20 = test_data['A_ACTIVATED'][:-1]
    x21 = test_data['A_REGISTERED'][:-1]
    x22 = test_data['O_DECLINED'][:-1]
    x23 = test_data['W_Wijzigen contractgegevens'][:-1]
    x_time = test_data['unix_rel_event_time'][1:]
    y_train = test_data['event concept:name'][1:]

    zipped = zip(x_time, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21,
                 x22, x23)

    X_test = [list(a) for a in zipped]
    return X_test


def fit_tree(X, y):
    boom = tree.DecisionTreeClassifier()
    boom.fit(X, y)

    return boom