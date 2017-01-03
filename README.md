# Project B2

Aim of this project is to perform prediction on a test set of genes and provide an evaluation using Viterbi decoding algorithm and supervised learning.

### Data

Data that we used to test our algorithms were provided in *sacCer3.genes.filtered.txt* file. These data represent known genes as well as exons and introns of the *Saccharomyces Cerevisiae* organism.

### Methods

#### HMM
In our Hidden Markov Model we have 2 hidden states: intron and exon and emissions A, T, C and G. 
#### Viterbi learning
We will initialize our matrices to some random values which sum to 1 and now using the HMM model and the emitted sequence we will start training our model.
#### Supervised learning
In supervised learning we will use statistics of known samples, that is the number of transitions from exon to intron and number of emissions of A, T, C and G at some state. 

### Results
First of all we divided our data to training and testing sets. We trained our model using 70% of input data. Later on trained model was tested using the rest of data (30%). We obtained 0.473% of accuracy.
```sh
Dataset = read_data()
testing_percentage = 0.3
test_data = Dataset[0:round(testing_percentage*len(Dataset))]
learning_data = Dataset[len(test_data):len(Dataset)]
hmm.supervised_learning(learning_data)
```
Second training scenario was based on crossvalidation approach. Taking one testing sample, we trained model using the rest of data and repeated this operation on all of testing samples. Than we took the average of all tests. Using crossvalidation approach accuracy of our algorithm turned out to be 46%.

```sh
accuracies = []
for  k, example in enumerate(Dataset):
    hmm.supervised_learning(Dataset[:k] + Dataset[(k + 1):])
accuracies.append(hmm.testing([example]))
```
We performed Viterbi learning on every training example to get the new emission and transition matrices which we will test on other examples. Test have shown that with this method we can obtain only 25% of accuracy.

```sh
Dataset = read_data()
testing_percentage = 0.3
test_data = Dataset[0:round(testing_percentage*len(Dataset))]
learning_data = Dataset[len(test_data):len(Dataset)]
hmm.supervised_learning_beta(learning_data)
```
