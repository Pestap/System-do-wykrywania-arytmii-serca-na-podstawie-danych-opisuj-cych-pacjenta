from keras.callbacks import EarlyStopping
from tensorflow import keras

from importAndPrepareData import prepareData
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, Conv1D, Bidirectional, MaxPooling1D

import numpy as np
import matplotlib.pyplot as plt


(X_train, Y_train), (X_test, Y_test) = prepareData("arrhythmia.data")

model = Sequential()
#model.add(Dense(32, activation='relu', input_dim= X_train.shape[1]))
#model.add(Dropout(rate=0.75))
#model.add(Dense(1, activation='sigmoid'))

#reshape X_train

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


model.add(Conv1D(filters = 128, kernel_size=25, activation='relu', input_shape= (278, 1)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode ='min', verbose = 1, patience=50, restore_best_weights=True)
#model.add(Bidirectional(LSTM(50, input_shape=(278,1))))
#model.add(Dense(1, activation='sigmoid'))

opt = keras.optimizers.Adam(learning_rate=0.0001)

model.compile(
    loss = 'binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

history = model.fit(X_train, Y_train, validation_split=0.2,batch_size=32, epochs=1500,callbacks=es, verbose = 1)

scores = model.evaluate(X_test, Y_test, verbose=0)

print(str(scores[1]))
plt.plot(history.history['loss'], label="Loss")
plt.plot(history.history['val_loss'], label="Validation loss")
plt.legend()
plt.show()

