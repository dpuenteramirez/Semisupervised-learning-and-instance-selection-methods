#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    RNN.py
# @Author:      Daniel Puente Ramírez
# @Time:        22/11/21 08:22

from collections import deque

import numpy as np

from CNN import CNN


def RNN(X):
    """
    Implementation of The Reduced Nearest Neighbor

    RNN is an extension of CNN. Firstly CNN will be executed in order to have
    S-CCN. It will perform iterative sample removal from S, and reclasificate
    all T, in hopes that there is no sample inside T classified incorrectly,
    in case there is at least one, the sample removed will be added again to S.

    :param X: dataset with scikit-learn structure.
    :return: the input dataset with the remaining samples.
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

    return S
