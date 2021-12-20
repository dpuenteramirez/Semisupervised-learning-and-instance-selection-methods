#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ENN_V2.py
# @Author:      Daniel Puente Ramírez
# @Time:        17/12/21 18:30
import copy

from sklearn.datasets import load_iris
import numpy as np
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
    S = copy.deepcopy(X)
    to_remove = []

    for index in range(size):
        x_sample = samples[index]
        other_samples = samples[:index] + samples[index + 1:]

        distances = []
        for dis_index in range(len(other_samples)):
            y_sample = samples[dis_index]
            y_target = targets[dis_index]
            distances.append([np.linalg.norm(x_sample - y_sample), y_target])

        distances.sort(key=lambda x: x[0])

        closest_clases = [x[1] for x in distances[:k]]

        counts = np.bincount(closest_clases)
        closest_class = np.argmax(counts)

        if closest_class != targets[index]:
            to_remove.append(index)

    S['data'] = np.delete(S['data'], to_remove, axis=0)
    S['target'] = np.delete(S['target'], to_remove, axis=0)

    return S


if __name__ == '__main__':
    iris = load_iris()
    print(f'Input samples: {len(iris.data)}')
    S = ENN(iris, 3)
    print(f'Output samples: {len(S.data)}')
    grafica_2D(S)
