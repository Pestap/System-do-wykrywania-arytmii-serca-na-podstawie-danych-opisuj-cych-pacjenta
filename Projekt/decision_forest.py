import os

import pandas as pd
import seaborn as sns

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from import_and_prepare_data import prepareData
import matplotlib.pyplot as plt


class DecisionForest:
    def __init__(self, trees=50):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.model = None
        self.labels = []
        self.classes = ['No arrhythmia', 'Arrhythmia']
        self.number_of_trees = trees
        self.accuracy = 0
        self.precision = 0
        self.recall = 0

    def import_data(self, filename):
        if os.path.exists("Data/"+filename):
            (self.X_train, self.Y_train), (self.X_test, self.Y_test), self.labels = prepareData(filename, False)
        else:
            raise Exception("No data file found")

    def construct_model(self):
        self.model = RandomForestClassifier(n_estimators=self.number_of_trees)

    def train_model(self):
        self.model.fit(self.X_train, self.Y_train)

    def test_model(self):
        self.predictions = self.model.predict(self.X_test)
        return metrics.accuracy_score(self.Y_test, self.predictions), metrics.precision_score(self.Y_test, self.predictions), metrics.recall_score(self.Y_test, self.predictions)

    def single_run(self):
        self.import_data('arrhythmia.data')
        self.construct_model()
        self.train_model()
        self.accuracy, self.precision, self.recall = self.test_model()

    def multiple_runs(self, n):
        accuracy = []
        precision = []
        recall = []

        for i in range(n):
            self.import_data('arrhythmia.data')
            self.construct_model()
            self.train_model()
            acc, prec, rec = self.test_model()
            accuracy.append(acc)
            precision.append(prec)
            recall.append(rec)

        self.accuracy = sum(accuracy) / len(accuracy)
        self.precision = sum(precision) / len(precision)
        self.recall = sum(recall) / len(recall)

    def print_result(self):
        print(f'Results for decision forest with {self.number_of_trees} trees:')
        print(f'Accuracy: {self.accuracy:.2f} %')
        print(f'Precision: {self.precision:.2f} %')
        print(f'Recall: {self.recall:.2f} %')

    def plot_result(self):
        sns.barplot(x=pd.Series(self.model.feature_importances_, index=self.labels).sort_values(ascending=False), y=self.labels)
        plt.xlabel("Feature Importance Score")
        plt.ylabel('Features', fontsize=2)
        plt.title("Feature Importance Visualization")
        plt.show()
