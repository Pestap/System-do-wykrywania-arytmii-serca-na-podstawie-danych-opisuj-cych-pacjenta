from keras.callbacks import EarlyStopping

from importAndPrepareData import prepareData
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, Conv1D, Bidirectional, MaxPooling1D

import numpy as np

np.random.seed(10)
(X_train, Y_train), (X_test, Y_test) = prepareData("arrhythmia.data")

model = Sequential()
#model.add(Dense(32, activation='relu', input_dim= X_train.shape[1]))
#model.add(Dropout(rate=0.25))
#model.add(Dense(1, activation='sigmoid'))

#reshape X_train

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


model.add(Conv1D(filters = 128, kernel_size=20, activation='relu', input_shape= (278, 1)))
model.add(Dropout(rate=0.8))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_accuracy', mode ='max', verbose = 1, patience=50)
#model.add(Bidirectional(LSTM(50, input_shape=(278,1))))
#model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss = 'binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(X_train, Y_train, validation_split=0.2,batch_size=32, epochs=1000,callbacks=es, verbose = 1)

scores = model.evaluate(X_test, Y_test, verbose=0)

print(str(scores[1]))

