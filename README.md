# Project B2

This project is for Introduction to Bioinformatics course at FRI University of Ljubljana 2016/17.  Aim of the project is to perform prediction on a test set of genes and provide an evaluation using supervised learning and unsupervised learning.
We defined an HMM model with two hidden states and observed characters which was further used to implement learning. The transition that we used was between exons (segments that end up in the protein) and introns (segments that are "spliced-out") on a given set of eukaryotic genes (we will do this in yeast - Saccharomyces Cerevisiae, which is maybe the simplest eukaryotic organism).

### Data

Data that we used to test our algorithms were provided in *sacCer3.genes.filtered.txt* file. These data represent known genes as well as exons and introns of the *Saccharomyces Cerevisiae* organism.
The test data consists of 329 sequences, where every sequence has the average length of 1794 nucleobasis. Each of the sequence contains at least one intron. For every sequence the path of introns and exons is provided. It is important especially for supervised learning. 

### Methods

#### HMM
A hidden Markov model (HMM) is one in which you observe a sequence of emissions, but do not know the sequence of states the model went through to generate the emissions. Analyses of hidden Markov models seek to recover the sequence of states from the observed data.
In our case we are going to have 2 hidden states S = {intron, exon}.
On this figure, we can see the example of hidden Markov model for our problem. 
The arrows that connects exons and introns represents probabilities of transitions and arrows connecting nucleobasis with introns and exons represents probabilities of emissions. 

![](http://imgur.com/ss50p8x.png)

#### Supervised learning
In supervised learning we used statistics of known samples, that is the number of transitions from exon to intron and number of emissions of A, T, C and G at some state. After counting all occurrences in all samples we calculated transition and emission matrices.
#### Unsupervised learning
In Viterbi learning we initialized our matrices to some random values which sum to 1 and using the HMM model and the emitted sequence we started training our model. We can use Viterbi decoding to find the most probable path and we considered the computed path and from it estimated the parameters of the model, emission and transition matrices. We repeated this procedure for all samples and we got final matrices which we tested.


### Results

#### Accuracy measures
We used two different accuracy measures: unweighted and weighted. Unweighted measure represents average of nucleobase-wised accuracies of all samples, where nucleobase-wised is ratio of correctly predicted introns and exons. Weighted accuracy represents weighted average of nucleobase-wised accuracies of all samples where weight of accuracy is equal to the length of given sequence.

#### Major classifier
Major classifier obtained unweighted accuracy of 75.3184130241% and weighted accuracy of 81.04262512788894%.
```sh
# Perform majority classifier (always predict exon)
accuracies = []
accuracies_weighted = []
normalization_sum = 0
for _, path in dataset:
    c_pred = path.count('e')
    accuracies.append(c_pred / len(path))
    accuracies_weighted.append(c_pred)
    normalization_sum += len(path)
    
print("Unweighted accuracy:", np.mean(accuracies) * 100, '[%]')
print("Weighted accuracy:", sum(accuracies_weighted) / normalization_sum * 100, '[%]')
```

#### Results of supervised learning
We performed two different test scenarios. With unweighted n-fold cross validation we obtained 75.2620292992% of accuracy and with weighted n-fold cross validation we obtained 84.86066034731586%.
```sh
# Perform n-fold cross validation
# Calculate weighted and unweighted accuracy
accuracies = []
accuracies_weighted = []
normalization_sum = 0
pred_pairs = []
for  k, example in enumerate(dataset):
    hmm.supervised_learning(dataset[:k] + dataset[(k + 1):])
    acc, pred = hmm.test([example], True)
    
    pred_pairs += list(zip(pred[0][0], pred[0][1]))
    accuracies.append(acc)
    accuracies_weighted.append(acc * len(example[0]))
    normalization_sum += len(example[0])
	
print("Unweighted accuracy:", np.mean(accuracies) * 100, '[%]')
print("Weighted accuracy:", sum(accuracies_weighted) / normalization_sum * 100, '[%]')
```
Here we have a plot of gene length and accuracy dependency:
![](http://i.imgur.com/5RS0QP4.png)

And confusion matrix is:
```sh
c = collections.Counter(pred_pairs)
print('\te\ti')
probs = [c[('e','e')] / (c[('e','e')] + c[('e','i')]),
         c[('e','i')] / (c[('e','e')] + c[('e','i')]),
         c[('i','e')] / (c[('i','e')] + c[('i','i')]),
         c[('i','i')] / (c[('i','e')] + c[('i','i')])]
probs = list(map(lambda x: str(round(x*100, 2)) + '%', probs))
print('e', probs[0], probs[1], sep='\t')
print('i', probs[2], probs[3], sep='\t')
```
|   | e | i |
| :------------ |:---------------:| -----:|
| e   | 94.98%	 |5.02% |
| i      | 58.4%      |  41.6% |
#### Results of unsupervised learning
We performed Viterbi learning on every training example to get the new emission and transition matrices which we will test on other examples. Test has shown that with this method we can obtain unweighted accuracy of 59.132641678556666% and weighted accuracy of 61.08856351083075%.
```sh
hmm.viterbi_learning_batch(dataset, iterations=5)
print("Unweighted accuracy:", hmm.test(dataset) * 100, '[%]')
print("Weighted accuracy:", hmm.test_weighted(dataset) * 100, '[%]')
print()
print(hmm)
```
### Knowledge mining
```sh
t, e = HMM.init_matrices_randomly('ei', 'ACTG')
hmm = HMM(t, e)
hmm.supervised_learning(dataset)
print(hmm)
```
From this we can conclude that:
- every gene starts with exon, 
- introns are three times shorter than exons,
- in introns we have higher ratio of T and A and in exons is the rest which supports a general knowledge that CG-content is higher in exons.
