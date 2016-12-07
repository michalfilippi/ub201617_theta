# Simple dataset_1 reader.
# Variable Dataset contains tuples (o, p), where o is a genome sequence and p
# is sequence of characters 'i'/'e' representing intron/exon

import os

Dataset = []

this_dir, _ = os.path.split(__file__)
dataset_path = os.path.join(this_dir, "dataset_1.txt")

with open(dataset_path, 'r') as f_in:
    observation = f_in.readline()
    path = f_in.readline()
    while path != '':
        Dataset.append((observation[:-1], path[:-1]))
        observation = f_in.readline()
        path = f_in.readline()

