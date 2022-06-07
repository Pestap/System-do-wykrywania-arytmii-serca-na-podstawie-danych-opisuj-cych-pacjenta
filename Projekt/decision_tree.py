import os

import graphviz
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from import_and_prepare_data import prepareData
import matplotlib.pyplot as plt


class DecisionTree:
    def __init__(self, depth):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.model = None
        self.predictions = None
        self.labels = []
        self.classes = ['No arrhythmia', 'Arrhythmia']
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.depth = depth

    def import_data(self, filename):
        if os.path.exists("Data/"+filename):
            (self.X_train, self.Y_train), (self.X_test, self.Y_test), self.labels = prepareData(filename, False)
        else:
            raise Exception("No data file found")

    def construct_model(self):
        self.model = DecisionTreeClassifier(criterion="gini", max_depth=self.depth)

    def train_model(self):
        self.model.fit(self.X_train, self.Y_train)

    def test_model(self):
        self.predictions = self.model.predict(self.X_test)
        return metrics.accuracy_score(self.Y_test, self.predictions), metrics.precision_score(self.Y_test, self.predictions), metrics.recall_score(self.Y_test, self.predictions)

    def single_run(self, feature_cols=None):
        self.import_data('arrhythmia.data')
        self.construct_model()
        self.train_model()
        self.accuracy, self.precision, self.recall = self.test_model()

    def multiple_runs(self,n):
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

        self.accuracy = sum(accuracy)/len(accuracy)
        self.precision = sum(precision)/len(precision)
        self.recall = sum(recall)/len(recall)

    def print_result(self):
        print(f'Results for decision tree with depth of {self.depth}')
        print(f'Accuracy: {self.accuracy:.2f} %')
        print(f'Precision: {self.precision:.2f} %')
        print(f'Recall: {self.recall:.2f} %')

    def plot_result(self):
        tree.plot_tree(self.model, filled=True, rounded=True, feature_names=self.labels, class_names=self.classes,
                       fontsize=8, proportion=True)
        plt.savefig("Plots/Decision_tree.png")

        plt.show()

    def save_result(self):
        dir = 'DecisionTreeVis'
        for f in os.listdir(dir):
            os.remove(os.path.join(dir,f))

        dot_data = tree.export_graphviz(self.model, out_file="DecisionTreeVis/arrhythmia_dec_tree.dot", feature_names=self.labels, class_names=self.classes,
                                        filled=True, rounded=True,special_characters=True)
        graph = graphviz.Source.from_file('DecisionTreeVis/arrhythmia_dec_tree.dot')
        graph.render('DecisionTreeVis/arrhythmia_dec_tree.gv', format='svg', view=False)
