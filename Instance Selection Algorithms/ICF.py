#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ICF.py
# @Author:      Daniel Puente Ramírez
# @Time:        23/11/21 09:37

from sklearn.datasets import load_iris
import numpy as np
from graficas import grafica_2D
from ENN import ENN


def __delete_multiple_element__(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)


def __coverage__(dat, tar):
    cov = np.zeros(len(dat))
    reachable = np.zeros(len(dat))

    for index in range(len(tar)):
        i1 = i2 = index
        target_class = tar[index]
        # Iterate Forward
        over = False
        while True:
            if i1 + 1 < len(tar) and not over:
                i1 += 1
            elif i1 + 1 == len(tar):
                over = True
                i1 = 0
            elif over:
                i1 += 1

            if tar[i1] == target_class:
                cov[index] += 1
                reachable[i1] += 1
            else:
                break

        # Iterate Backwards
        while True:
            i2 -= 1
            if tar[i2] == target_class:
                cov[index] += 1
                reachable[i2] += 1
            else:
                break

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
    while progress:

        coverage, reachable = __coverage__(data, target)

        progress = False
        remove_indexes = []
        for index, instance in enumerate(zip(data, target)):
            (dat, tar) = instance
            if abs(reachable[index]) > abs(coverage[index]):
                remove_indexes.append(index)
                progress = True
        print(progress)
        __delete_multiple_element__(data, remove_indexes)
        __delete_multiple_element__(target, remove_indexes)

    S['data'] = data
    S['target'] = target
    print(f"{len(X['data']) - len(S['data'])} samples deleted.")
    return S


def main():
    data = load_iris()

    S = ICF(X=data)

    #grafica_2D(S)


if __name__ == '__main__':
    main()