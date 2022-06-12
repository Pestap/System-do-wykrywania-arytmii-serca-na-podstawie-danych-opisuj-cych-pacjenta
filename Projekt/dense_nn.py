import os.path
import time

import keras.models
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from sklearn import metrics
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
from import_and_prepare_data import prepareData


class DenseNN:
    def __init__(self):
        self.history = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.model = None
        self.predictions = None

    def import_data(self, filename):
        if os.path.exists("Data/"+filename):
            (self.X_train, self.Y_train), (self.X_test, self.Y_test), _ = prepareData(filename)
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

    def train_model(self,verbose=0):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=1000, restore_best_weights=True)
        self.history = self.model.fit(self.X_train, self.Y_train, validation_split=0.2, batch_size=32, epochs=10000, callbacks=es, verbose=verbose)
        np.save('Models/history/denseNN_history.npy', self.history)
        self.model.save("Models/denseNN")

    def test_model(self):
        if self.model == None:
            self.model = keras.models.load_model('Models/denseNN')

        self.scores = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        self.predictions = self.model.predict(self.X_test)
        self.predictions = list(map(lambda x: 0 if x<0.5 else 1, self.predictions))

    def get_results(self, verbose=1):

        # TODO: get all resutlts: accuracy, loss, precision etc.
        plt.clf()
        plt.cla()

        plt.plot(self.history.history['accuracy'], label="Accuracy")
        plt.plot(self.history.history['val_accuracy'], label="Validation accuracy")
        plt.legend()
        plt.grid()
        plt.title("Własności modelu")
        plt.xlabel("Epoki")
        plt.ylabel("Celność")
        curr_time = time.time()
        plt.savefig(f'Plots/model_{curr_time}_acc.png')

        plt.clf()
        plt.cla()
        plt.plot(self.history.history["loss"], label="Loss")
        plt.plot(self.history.history['val_loss'], label="Validation loss")
        plt.legend()
        plt.grid()
        plt.title("Własności modelu")
        plt.xlabel("Epoki")
        plt.ylabel("Wartość funkcji loss")
        plt.savefig(f'Plots/model_{curr_time}_loss.png')

        if verbose == 1:
            plt.show()

        return self.scores[0], self.scores[1], metrics.precision_score(self.Y_test, self.predictions), metrics.recall_score(self.Y_test, self.predictions)

    def single_run(self, verbose=1):
        start = time.time()
        self.import_data('arrhythmia.data')
        self.construct_model()
        self.train_model(verbose)
        self.test_model()
        loss, acc, precision, recall = self.get_results(verbose)
        stop = time.time()
        if verbose == 1:
            print("Scores on test set:")
            print(f"Time: {(stop-start):.2f} s")
            print(f"Loss: {loss:.2f}")
            print(f"Accuracy: {(acc*100):.2f} %")
            print(f"Precision: {(precision * 100):.2f} %")
            print(f"Recall: {(recall * 100):.2f} %")
        return loss, acc,precision,recall, stop-start

    def multiple_runs(self, runs, verbose=0):
        avg_loss = []
        avg_acc = []
        avg_precision = []
        avg_recall = []
        for i in range(runs):
            loss, acc,precision, recall, run_time = self.single_run(verbose)
            avg_loss.append(loss)
            avg_acc.append(acc)
            avg_precision.append(precision)
            avg_recall.append(recall)
            print(f"Run {i+1} finished ({run_time:.2f} s, accuracy: {(acc*100):.2f} %, precision: {(precision*100):.2f} %, recall {(recall*100):.2f} %) - {runs-i-1} remaining.")

        print(f"Average results of {runs} runs:")
        print(f"Loss: {(sum(avg_loss)/len(avg_loss)):.2f}")
        print(f"Accuracy: {(sum(avg_acc)/len(avg_acc) * 100):.2f}%")
        print(f"Precision: {(sum(avg_precision)/len(avg_precision) * 100):.2f}%")
        print(f"Recall: {(sum(avg_recall) / len(avg_recall) * 100):.2f}%")