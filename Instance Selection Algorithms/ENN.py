#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ENN.py
# @Author:      Daniel Puente Ramírez
# @Time:        16/11/21 17:14
import copy
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


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
    S1 = copy.deepcopy(X)
    borrado = 0
    for index, flower in enumerate(X.data):
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
            borrado += 1
            S1['target'][index - 1] = 3
            np.delete(X.data, index - 1, 0)
            np.delete(X.target, index - 1, 0)

    print(f"Se han eliminado {borrado} muestras.")
    return X, S1


def main():
    data = load_iris()
    k = 3
    target_names = list(data.target_names)
    target_names.append('deleted')
    data['target_names'] = np.array(target_names)

    S, S1 = ENN(X=data, k=k)

    # https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_iris_scatter.html
    x_index = 0
    y_index = 1
    formatter = plt.FuncFormatter(lambda i, *args: S1.target_names[int(i)])
    plt.figure(figsize=(5, 4))
    plt.scatter(S1.data[:, x_index], S1.data[:, y_index], c=S1.target)
    plt.colorbar(ticks=[0, 1, 2, 3], format=formatter)
    plt.xlabel(S1.feature_names[x_index])
    plt.ylabel(S1.feature_names[y_index])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
