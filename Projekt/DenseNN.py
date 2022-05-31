import os.path

import keras.models
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
from importAndPrepareData import prepareData


class DenseNN:
    def __init__(self):
        self.history = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.model = None

    def import_data(self, filename):
        if os.path.exists("Data/"+filename):
            (self.X_train, self.Y_train), (self.X_test, self.Y_test) = prepareData(filename)
        else:
            raise Exception("No data file found")

    def construct_model(self):
        self.model = Sequential()
        self.model.add(Dense(self.X_train.shape[1], activation='relu', input_dim=self.X_train.shape[1]))
        self.model.add(Dropout(rate=0.6))
        self.model.add(Dense(1, activation='sigmoid'))

        optimizer = keras.optimizers.Adam(learning_rate=0.0001)

        self.model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

    def train_model(self):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1000, restore_best_weights=True)
        self.history = self.model.fit(self.X_train, self.Y_train, validation_split=0.2, batch_size=32, epochs=10000, callbacks=es, verbose=1)
        np.save('Models/history/denseNN_history.npy', self.history)
        self.model.save("Models/denseNN")

    def test_model(self):
        if self.model == None:
            self.model = keras.models.load_model('Models/denseNN')

        self.scores = self.model.evaluate(self.X_test, self.Y_test, verbose=0)

    def get_results(self):
        print(f"Scores on test set:")
        print(f"loss: {self.scores[0]}")
        print(f"accuracy: {self.scores[1]}")
        plt.plot(self.history.history["loss"], label="Loss")
        plt.plot(self.history.history['val_loss'], label="Validation loss")
        plt.plot(self.history.history['accuracy'], label="Accuracy")
        plt.plot(self.history.history['val_accuracy'], label="Validation accuracy")
        plt.legend()
        plt.grid()
        plt.title("Własności modelu")
        plt.xlabel("Epoki")
        plt.ylabel("Wartość")
        plt.show()
        plt.savefig("model.png")

        return self.scores[0], self.scores[1]