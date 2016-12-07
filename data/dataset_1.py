import numpy as np

# Simple dataset_1 reader.
# Variable Dataset contains tuples (o, p), where o is a genome sequence and p
# is sequence of characters 'i'/'e' representing intron/extron


Dataset = []

with open("dataset_1.txt", 'r') as f_in:
    observation = f_in.readline()
    path = f_in.readline()
    while path != '':
        Dataset.append((observation[:-1], path[:-1]))
        observation = f_in.readline()
        path = f_in.readline()


print(len(Dataset))
