import os

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from importAndPrepareData import prepareData
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus


class DecisionTree:
    def __int__(self):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.model = None
        self.predictions

    def import_data(self, filename):
        if os.path.exists("Data/"+filename):
            (self.X_train, self.Y_train), (self.X_test, self.Y_test) = prepareData(filename, False)
        else:
            raise Exception("No data file found")

    def construct_model(self):
        self.model = DecisionTreeClassifier(criterion="gini", max_depth=7)

    def train_model(self):
        self.model.fit(self.X_train, self.Y_train)

    def test_model(self):
        self.predictions = self.model.predict(self.X_test)
        print(f"Acc: {metrics.accuracy_score(self.Y_test, self.predictions)}")
        return metrics.accuracy_score(self.Y_test, self.predictions)

    def single_run(self, feature_cols=None):
        self.import_data('arrhythmia.data')
        self.construct_model()
        self.train_model()
        result = self.test_model()
        tree.plot_tree(self.model, filled=True, rounded=True, class_names=['No arrhythmia', 'Arrhythmia'], fontsize=8,proportion=True)
        #plt.show()
        return result