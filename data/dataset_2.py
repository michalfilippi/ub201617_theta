# Simple dataset_1 reader.
# Variable Dataset contains tuples (o, p), where o is a list of base triplets and p
# is sequence of characters 'i'/'e' representing intron/exon

import os
import collections

def read_data():

    Dataset = []

    this_dir, _ = os.path.split(__file__)
    dataset_path = os.path.join(this_dir, "dataset_1.txt")

    with open(dataset_path, 'r') as f_in:
        observation = f_in.readline()
        path = f_in.readline()
        while path != '':
            observation = observation[:-1]
            observation_triplets = []
            window = collections.deque(['X', observation[0]], maxlen=3)
            for c in observation[1:]:
                window.append(c)
                observation_triplets.append(''.join(window))
            window.append('X')
            observation_triplets.append(''.join(window))

            Dataset.append((observation_triplets, path[:-1]))
            observation = f_in.readline()
            path = f_in.readline()

    return Dataset
