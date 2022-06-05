from DenseNN import DenseNN
from DecisionTree import DecisionTree

nn_model = DenseNN()
decision_tree = DecisionTree()
scores = []
for i in range(20):
    scores.append(decision_tree.single_run())

print(str(sum(scores)/len(scores)))

nn_model.single_run()

#nn_model.multiple_runs(1)
#nn_model.single_run()