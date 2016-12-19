from lib.markov_models import HMM
from data.dataset_1 import read_data
import numpy as np

a = {'i': {'i': 1.0, 'e': 0.0}, 'e': {'i': 0.5, 'e': 0.5}}
b = {'i': {'C': 0.20679611650485438, 'A': 0.3087378640776699, 'T': 0.2854368932038835, 'G': 0.19902912621359223}, 'e': {'C': 0.25, 'A': 0.25, 'T': 0.25, 'G': 0.25}}


hmm = HMM(a,b)

Dataset = read_data()
testing_percentage = 0.3
test_data = Dataset[0:round(testing_percentage*len(Dataset))]
learning_data = Dataset[len(test_data):len(Dataset)]
hmm.supervised_learning(learning_data)
print(hmm.testing(test_data))

# crossvalidation
accuracies = []
for  k, example in enumerate(Dataset):
    hmm.supervised_learning(Dataset[:k] + Dataset[(k + 1):])
    accuracies.append(hmm.testing([example]))

print(accuracies)
print (np.mean(accuracies))
