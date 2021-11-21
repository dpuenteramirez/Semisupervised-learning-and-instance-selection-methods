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

    For each flower locates the *k* neighbors and selects the number of different classes there are.
    It has careful with the last values of the data, so if it reaches the end of the array, starts
    looking at the start of himself.
    :param X:
    :param k:
    :return:
    """
    y = X.target
    deleted_samples = 0
    for index in range(len(X['data'])):
        classes = {}
        index += 1
        neighbors_classes = y[index:index + k] if len(y[index:index + k]) == k else np.hstack(
            (y[index:index + k], y[0:k - len(y[index:index + k])]))
        for neigh in neighbors_classes:
            try:
                classes[neigh] += 1
            except KeyError:
                classes[neigh] = 0
        if max(classes, key=classes.get) != y[index - 1]:
            deleted_samples += 1
            np.delete(X.data, index - 1, 0)
            np.delete(X.target, index - 1, 0)

    print(f"{deleted_samples} samples deleted.")
    return X


def main():
    data = load_iris()
    k = 3

    S = ENN(X=data, k=k)

    grafica_2D(S)


if __name__ == '__main__':
    main()
