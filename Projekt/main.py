from matplotlib import pyplot as plt

from dense_nn import DenseNN
from decision_forest import DecisionForest
from decision_tree import DecisionTree
nn_model = DenseNN()
decision_forest = DecisionForest(100)
decision_tree = DecisionTree(10)

decision_forest.multiple_runs(70)
decision_forest.print_result()
decision_forest.plot_result()

#decision_tree.single_run()
#decision_tree.print_result()
#decision_tree.plot_result()
#decision_tree.save_result()

#nn_model.multiple_runs(200)
#nn_model.multiple_runs(1)
#nn_model.single_run()