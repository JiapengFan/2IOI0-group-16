import pandas as pd
from sklearn.model_selection import train_test_split


def dataSplitter(trainingData, testData):

    trainingData_grouped = trainingData.groupby(by=['case concept:name'])[
        'unix_abs_event_time'].max()
    testData_grouped = testData.groupby(by=['case concept:name'])[
        'unix_reg_time'].min()

    for index, value in testData_grouped.items():
        if (index in trainingData['case concept:name'].values):
            if (value < trainingData_grouped[index]):
                testData.drop(
                    testData.loc[testData['case concept:name'] == index].index, inplace=True)
                trainingData.drop(
                    trainingData.loc[trainingData['case concept:name'] == index].index, inplace=True)

    trainingData_per_case = trainingData.set_index(
        ["case concept:name", "eventID "])
    trainingData_per_case.sort_index()

    train_mask, validation_mask = train_test_split(
        trainingData_per_case.index.levels[0], test_size=0.1)

    train_dataset = trainingData_per_case.loc[train_mask].copy().sort_index()
    validation_dataset = trainingData_per_case.loc[validation_mask].copy(
    ).sort_index()

    train_dataset.reset_index(inplace=True)
    validation_dataset.reset_index(inplace=True)

    return (train_dataset, validation_dataset, testData)
