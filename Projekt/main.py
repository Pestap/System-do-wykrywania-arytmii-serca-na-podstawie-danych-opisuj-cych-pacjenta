from matplotlib import pyplot as plt

from dense_nn import DenseNN
from decision_forest import DecisionForest
from decision_tree import DecisionTree

nn_model = DenseNN()
decision_tree = DecisionTree(depth=25)
decision_forest = DecisionForest(trees=80, depth=25)

nn_model.single_run()

decision_tree.single_run()
decision_tree.print_result()
decision_tree.save_result()

decision_forest.single_run()
decision_forest.print_result()
decision_forest.plot_result()