#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    RNN.py
# @Author:      Daniel Puente Ramírez
# @Time:        22/11/21 08:22

from collections import deque
import numpy as np
from sklearn.datasets import load_iris
from CNN import CNN
from graficas import grafica_2D


def RNN(X):
    """

    :param X:
    :return:
    """
    S = CNN(X)
    data = deque([x for x in S['data']])
    target = deque([x for x in S['target']])

    for index, instance in enumerate(zip(S.data, S.target)):
        (sample, class_sample) = instance
        data.popleft()
        target.popleft()

        for x_class, x_sample in zip(X['target'], X['data']):
            euc = []
            for s_sample in data:
                euc.append(np.linalg.norm(s_sample - x_sample))
            euc = np.array(euc)
            euc_nn = np.amin(euc)
            index_nn = np.ravel(np.where(euc == euc_nn))
            nn_class = target[index_nn[0]]

            if nn_class != x_class:
                data.append(sample)
                target.append(class_sample)
                break

    S['data'] = np.array(data)
    S['target'] = target
    print(f"{len(data)} samples retrieved.")

    return S


def main():
    data = load_iris()
    S = RNN(X=data)
    grafica_2D(S)


if __name__ == '__main__':
    main()
