#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ENN.py
# @Author:      Daniel Puente Ramírez
# @Time:        16/11/21 17:14

import numpy as np
from sklearn.datasets import load_iris

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

    y = X.target
    data = []
    target = []
    for index in range(len(X['data'])):
        classes = {}
        index += 1

        if len(y[index:index + k]) == k:
            neighbors_classes = y[index:index + k]
        else:
            neighbors_classes = np.hstack((y[index:index + k],
                                           y[0:k - len(y[index:index + k])]))
        for neigh in neighbors_classes:
            try:
                classes[neigh] += 1
            except KeyError:
                classes[neigh] = 0

        if max(classes, key=classes.get) == y[index - 1]:
            data.append(X['data'][index])
            target.append(X['target'][index])

    X['data'] = np.array(data)
    X['target'] = target

    return X


def main():
    data = load_iris()
    n_samples = len(data['data'])
    k = 3
    S = ENN(X=data, k=k)

    print(f"{n_samples - len(S['data'])} samples deleted.")
    grafica_2D(S)


if __name__ == '__main__':
    main()
