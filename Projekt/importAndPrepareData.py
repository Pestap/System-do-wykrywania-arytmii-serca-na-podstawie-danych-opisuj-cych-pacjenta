import pandas as pd
import numpy as np

def importData(filename):
    #import z pliku z danymi z folderu Data
    df = pd.read_csv("Data/" + filename, header=None)
    return df


def prepareData(filename):
    df = importData(filename)

    dataset = df.to_numpy()
    #usuwamy 13 kolumne - większość rekordów nie posiada wartości
    dataset = np.delete(dataset, 13, 1)

    #usuwamy rekordy nieposiadjace plenych atrybutow
    rows_to_delete = []
    for idx, row in enumerate(dataset):
        if '?' in row:
            rows_to_delete.append(idx)

    dataset = np.delete(dataset, rows_to_delete, 0)

    dataset = dataset.astype(float)

    training_set_size = int(0.8 * dataset.shape[0])
    test_set_size = np.shape(dataset)[0] - training_set_size

    #losujemy zbiór testowy i treningowy: proporcje 20 : 80
    training_set_indexes = np.random.choice(dataset.shape[0], size=training_set_size, replace=False)
    all_indexes = [ x for x in range(dataset.shape[0])]
    test_set_indexes = np.setdiff1d(all_indexes, training_set_indexes)

    #training_set_indexes.sort()
    #test_set_indexes.sort()

    training_set = dataset[training_set_indexes,:]
    test_set = dataset[test_set_indexes, :]


    #dzielimy na zbiór x i y

    training_set_Y = training_set[:,-1]
    test_set_Y = test_set[:,-1]

    for i in range(training_set_Y.shape[0]):
        if training_set_Y[i] == 1:
            training_set_Y[i] = 0
        elif training_set_Y[i] > 1:
            training_set_Y[i] = 1

    for i in range(test_set_Y.shape[0]):
        if test_set_Y[i] == 1:
            test_set_Y[i] = 0
        elif test_set_Y[i] > 1:
            test_set_Y[i] = 1

    training_set = np.delete(training_set, -1, 1)
    test_set = np.delete(test_set, -1, 1)

    return (training_set, training_set_Y),(test_set, test_set_Y)

