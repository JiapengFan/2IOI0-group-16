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
    y = dummy_data['event concept:name']


def fit_tree(X, y):
    boom = tree.DecisionTreeClassifier()
    boom.fit(X, y)

    return boom