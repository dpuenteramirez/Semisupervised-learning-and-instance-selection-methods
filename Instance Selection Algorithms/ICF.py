#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ICF.py
# @Author:      Daniel Puente Ramírez
# @Time:        23/11/21 09:37

import sys

import numpy as np
from sklearn.datasets import load_iris

from ENN import ENN
from graficas import grafica_2D


def __delete_multiple_element__(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)


def __coverage__(dat, tar):
    cov = [[None] for _ in range(len(dat))]
    reachable = [[None] for _ in range(len(dat))]

    matrix_distances = [[sys.maxsize for _ in range(len(dat))]
                        for _ in range(len(dat))]

    for index in range(len(tar)):
        for index2 in range(index, len(tar)):
            euc = np.linalg.norm(dat[index] - dat[index2])
            matrix_distances[index][index2] = euc
            matrix_distances[index2][index] = euc

    closest_enemy = [[None, None] for _ in range(len(dat))]
    for index, row in enumerate(matrix_distances):
        x_class = tar[index]
        enemies = np.where(tar != x_class)[0]
        closest_enemy[index] = [enemies[0], matrix_distances[index][enemies[0]]]
        for enemy in enemies[1:]:
            if matrix_distances[index][enemy] < closest_enemy[index][1]:
                closest_enemy[index] = [enemy, matrix_distances[index][enemy]]

    for index, row in enumerate(matrix_distances):
        x_class = tar[index]
        allies = np.where(tar == x_class)[0]
        for ally in allies:
            if matrix_distances[index][ally] < closest_enemy[index][1]:
                cov[index].append(ally)
                reachable[ally].append(index)
    cov = [[i for i in val if i] for val in cov]
    reachable = [[i for i in val if i] for val in reachable]

    return cov, reachable


def ICF(X):
    """

    :param X:
    :return:
    """
    # Perform Wilson Editing
    S = ENN(X=X, k=3)

    # Iterate until no cases flagged for removal
    data = S['data']
    target = S['target']
    progress = True
    removed_samples = 0
    while progress:

        coverage, reachable = __coverage__(data, target)

        progress = False
        remove_indexes = []
        for index, instance in enumerate(zip(data, target)):
            (dat, tar) = instance
            if abs(len(reachable[index])) > abs(len(coverage[index])):
                remove_indexes.append(index)
                progress = True
                removed_samples += 1
        __delete_multiple_element__(data, remove_indexes)
        __delete_multiple_element__(target, remove_indexes)

    S['data'] = np.array(data)
    S['target'] = target

    print(f"{removed_samples} samples deleted.")
    return S


def main():
    data = load_iris()
    S = ICF(X=data)
    grafica_2D(S)


if __name__ == '__main__':
    main()
