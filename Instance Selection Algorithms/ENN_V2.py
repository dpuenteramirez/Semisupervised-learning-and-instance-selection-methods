#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ENN_V2.py
# @Author:      Daniel Puente Ramírez
# @Time:        17/12/21 18:30


from sklearn.datasets import load_iris
import numpy as np
from sklearn.neighbors import NearestNeighbors
from graficas import grafica_2D


def ENN(X, k):
    """
    Implementation of the Wilson Editing algorithm.

    For each sample locates the *k* nearest neighbors and selects the number of
    different classes there are.
    If a sample results in a wrong classification after being classified with
    k-NN, that sample is removed from the TS.
    :param X: dataset with scikit-learn structure.
    :param k: int: number of neighbors to evaluate.
    :return: the input dataset with the remaining samples.
    """
    size = len(X['data'])
    samples = list(X['data'])
    targets = X['target']

    removed = 0
    index = 0
    while True:
        other_samples = samples[:index] + samples[index+1:]
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(
            other_samples)
        distances, indices = nbrs.kneighbors([samples[index]])
        closest_clases = targets[indices[0]]
        counts = np.bincount(closest_clases)
        closest_class = np.argmax(counts)

        if closest_class != targets[index]:
            del samples[index]
            targets = np.delete(targets, index)
            removed += 1
        else:
            index += 1

        if index+removed == size:
            break
    X['data'] = np.array(samples)
    X['target'] = targets

    return X


if __name__ == '__main__':
    iris = load_iris()
    print(f'Input samples: {len(iris.data)}')
    S = ENN(iris, 3)
    print(f'Output samples: {len(S.data)}')
    grafica_2D(S)