from DenseNN import DenseNN
nn_model = DenseNN()

losses = []
accs = []

for i in range(10):
    nn_model.import_data('arrhythmia.data')
    nn_model.construct_model()
    nn_model.train_model()
    nn_model.test_model()
    loss, acc = nn_model.get_results()
    losses.append(loss)
    accs.append(acc)

print(sum(losses)/len(losses))
print(sum(accs)/len(accs))