from keras.callbacks import EarlyStopping
from tensorflow import keras

from importAndPrepareData import prepareData
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, Conv1D, Bidirectional, MaxPooling1D

import numpy as np
import matplotlib.pyplot as plt

scoresGlobal = []
droputRates = []

for i in range(9):
    scoresForDropout = []
    for j in range(10):
        (X_train, Y_train), (X_test, Y_test) = prepareData("arrhythmia.data")



        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim= X_train.shape[1]))

        model.add(Dropout(rate=0.1 + i*0.1))
        model.add(Dense(1, activation='sigmoid'))


        es = EarlyStopping(monitor='val_loss', mode ='min', verbose = 0,min_delta=0.02, patience=100, restore_best_weights=True)
        #model.add(Bidirectional(LSTM(50, input_shape=(278,1))))
        #model.add(Dense(1, activation='sigmoid'))

        opt = keras.optimizers.Adam(learning_rate=0.0001)

        model.compile(
            loss = 'binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )

        history = model.fit(X_train, Y_train, validation_split=0.2, batch_size=32, epochs=1500, callbacks=es, verbose = 0)

        scores = model.evaluate(X_test, Y_test, verbose=0)

        print(str(scores[1] * 100) + " % accuracy on test set")
        scoresForDropout.append(scores[1])

    droputRates.append(0.1 + i * 0.1)
    scoresGlobal.append(scoresForDropout)
    print("Average score for dropout " + str(0.1 + i*0.1) +": " + str(np.average(scoresForDropout)) +"%")
print(scoresGlobal)

plt.clf()
averageScores = np.average(scoresGlobal, axis =1)
plt.plot(droputRates, averageScores)
plt.grid()
plt.show()
#TODO: niech tworzy nowy plik z sklasyfikowanymi danymi
