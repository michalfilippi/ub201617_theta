from lib.markov_models import HMM
from data.dataset_1 import read_data


# a = {"i": {"i": 0.582, "e": 0.481}, "e": {"i": 0.272, "e": 0.728}} #transition matrix
# b = {"i": {"A": 0.129, "C": 0.35, "T": 0.38, "G": 0.14}, "e": {"A": 0.42, "C": 0.11, "T": 0.23, "G": 0.24}} #emision matrix

a = {'i': {'i': 1.0, 'e': 0.0}, 'e': {'i': 0.5, 'e': 0.5}}
b = {'i': {'C': 0.20679611650485438, 'A': 0.3087378640776699, 'T': 0.2854368932038835, 'G': 0.19902912621359223}, 'e': {'C': 0.25, 'A': 0.25, 'T': 0.25, 'G': 0.25}}

# a = {"i": {"i": 0.582, "e": 0.418}, "e": {"i": 0.272, "e": 0.728}} #transition matrix
# b = {"i": {"x": 0.129, "y": 0.35, "z": 0.52}, "e": {"x": 0.422, "y": 0.151, "z": 0.426}} #emision matrix


# sequence = "xxxzyzzxxzxyzxzxyxxzyzyzyyyyzzxxxzzxzyzzzxyxzzzxyzzxxxxzzzxyyxzzzzzyzzzxxzzxxxyxyzzyxzxxxyxzyxxyzyxz"

hmm = HMM(a,b)


Dataset = read_data()
testing_percentage = 0.3
test_data = Dataset[0:round(testing_percentage*len(Dataset))]
learning_data = Dataset[len(test_data):len(Dataset)]
hmm.supervised_learning(learning_data)

print(hmm.testing(test_data))