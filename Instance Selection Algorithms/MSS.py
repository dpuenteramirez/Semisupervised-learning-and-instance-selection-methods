#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    MSS.py
# @Author:      Daniel Puente Ramírez
# @Time:        23/11/21 11:53
import sys

import numpy as np
from sklearn.datasets import load_iris

from graficas import grafica_2D


def __enemy_distance__(dat, tar):
    """
    Inner method which calculates the distance between each sample and its
    nearest enemy.
    :param dat: T samples.
    :param tar: T classes.
    :return: list of triplets: [sample, class, distance], sorted by the
    smallest distance.
    """
    solution = []
    for sample, x_class in zip(dat, tar):
        distance = sys.maxsize
        for sample_1, x1_class in zip(dat, tar):
            if x1_class == x_class:
                continue
            else:
                euc = np.linalg.norm(sample - sample_1)
                if euc < distance:
                    distance = euc
        solution.append([sample, x_class, distance])

    solution.sort(key=lambda x: x[2])

    return solution


def MSS(X):
    """
    Implementation of Modified Selective Subset

    It starts with two empty arrays *dat* and *tar*, which will contain the
    instances selected. The first approach is to sort based on Dj all the
    dataset. And after that it will perform for every xi, check if xj is in
    X, if so it will check d(xi,xj) < Dj, if so will mark xj for deletion.
    This deletion process is not necessary due to *remove_indexes* which
    keeps track of all indexes marked for removal. If some removed had to be
    done, the *add* boolean will be set to True and it will result in xi
    being added to the results dataset.
    :param X: dataset with scikit-learn structure.
    :return: the input dataset with the remaining samples.
    """

    triplets = __enemy_distance__(X['data'], X['target'])
    dat = []
    tar = []
    remove_indexes = []

    for index in range(len(triplets)):
        if index in remove_indexes:
            continue
        (sample, x_class, distance) = triplets[index]
        add = False
        for index1 in range(index, len(triplets)):
            if index1 in remove_indexes:
                continue
            if np.linalg.norm(sample - triplets[index1][0]) < distance:
                remove_indexes.append(index1)
                add = True

        if add:
            dat.append(sample)
            tar.append(x_class)

    X['data'] = np.array(dat)
    X['target'] = tar
    return X


def main():
    data = load_iris()
    n_samples = len(data['data'])
    S = MSS(X=data)

    print(f"{n_samples - len(S['data'])} samples deleted.")
    grafica_2D(S)


if __name__ == '__main__':
    main()
