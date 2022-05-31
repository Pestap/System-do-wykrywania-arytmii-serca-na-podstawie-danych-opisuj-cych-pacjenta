import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

def importData(filename):
    #import z pliku z danymi z folderu Data
    df = pd.read_csv("Data/" + filename, header=None)
    return df

def column_average(dataset, column_index):
    column = dataset[:, column_index]
    sum = 0
    count = column.shape[0]

    for val in column:
        if val != '?':
            sum += int(val)
        else:
            count -= 1

    return sum/count


def find_max_in_column(column):
    max_idx = 0
    max_val = abs(column[0])

    for val in column:
        if val > max_val:
            max_val = val

    return max_val


def normalize_dataset(dataset):
    maximum_values = np.amax(np.abs(dataset), axis=0)
    means = np.mean(dataset, axis=0)
    std = np.std(dataset, axis=0)

    normalization_result = np.zeros(dataset.shape)

    for row_idx, row in enumerate(dataset):
        for val_idx, val in enumerate(row):
            normalization_result[row_idx][val_idx] = (dataset[row_idx][val_idx] - means[val_idx])/std[val_idx]



    return normalization_result

def prepareData(filename):
    df = importData(filename)

    dataset = df.to_numpy()
    #usuwamy 13 kolumne - większość rekordów nie posiada wartości
    cols_to_delete = [13]
    #oznaczmy jako do usunięcia kolumnny, które we wszytkich mają 0
    t_dataset = np.transpose(dataset)

    for col_idx, col in enumerate(t_dataset):
        amount_of_zeros =0
        for val in col:
            if val == 0:
                amount_of_zeros += 1
            if val == '?':
                cols_to_delete.append(col_idx)
                break
        if amount_of_zeros/len(col) >= 0.95:
            cols_to_delete.append(col_idx)

    #pozbywamy się ewentualnych duplikatów
    cols_to_delete = set(cols_to_delete)
    cols_to_delete = list(cols_to_delete)


    dataset = np.delete(dataset, cols_to_delete, 1)

    #usuwamy rekordy nieposiadjace plenych atrybutow
    #rows_to_delete = []

    for idx, row in enumerate(dataset):
        for val_idx, val in enumerate(row):
            if val == '?':
                dataset[idx][val_idx] = column_average(dataset, val_idx)

    dataset = dataset.astype(float)

    training_set_size = int(0.84 * dataset.shape[0])
    test_set_size = np.shape(dataset)[0] - training_set_size

    #losujemy zbiór testowy i treningowy: proporcje 16: 84
    training_set_indexes = np.random.choice(dataset.shape[0], size=training_set_size, replace=False)
    all_indexes = [ x for x in range(dataset.shape[0])]
    test_set_indexes = np.setdiff1d(all_indexes, training_set_indexes)


    training_set = dataset[training_set_indexes,:]
    test_set = dataset[test_set_indexes, :]


    #scaler = StandardScaler()
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

    scaler = StandardScaler()
    scaler.fit(training_set)

    training_set = scaler.transform(training_set)
    test_set = scaler.transform(test_set)

    return (training_set, training_set_Y), (test_set, test_set_Y)

