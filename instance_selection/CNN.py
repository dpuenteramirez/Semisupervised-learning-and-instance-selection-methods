#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    CNN.py
# @Author:      Daniel Puente Ramírez
# @Time:        19/11/21 07:13

import numpy as np


def __delete_multiple_element__(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)


def CNN(X):
    """
    Implementation of The Condensed Nearest Neighbor Rule

    The first sample of each class is placed in *store*. Thus we only have
    one the second sample classification will be trivial. When a sample is
    classified correctly it is places in *handbag*, otherwise it is placed
    inside *store*. This procedure will be repeated until the nth sample is
    being classified, whether it was correctly or not.
    Once the first loop is finished, it continues looping through *handbag*
    until termination, what will happen if either of these two cases occurs:
    - All the samples have been added to *store*, which means *handbag* is
    empty.
    - One complete pass has been made through *handbag* and none samples
    where transferred to *store*. The following passes will end up with the
    same result as the underlying decision surface has not been changed.
    Extracted from:
    The condensed nearest neighbor rule. IEEE Transactions on Information
    Theory ( Volume: 14, Issue: 3, May 1968)
    :param X: dataset with scikit-learn structure.
    :return: the input dataset with the remaining samples.
    """

    store_classes, indexes = np.unique(X.target, return_index=True)
    store_classes = store_classes.tolist()
    store = [X['data'][x] for x in indexes]

    handbag = []

    for sample_class, sample in zip(X.target, X.data):
        euc = []
        for s in store:
            euc.append(np.linalg.norm(s - sample))
        euc = np.array(euc)
        euc_nn = np.amin(euc)
        index_nn = np.ravel(np.where(euc == euc_nn))
        nn_class = store_classes[index_nn[0]]

        if nn_class == sample_class:
            handbag.append((sample_class, sample))
        else:
            store.append(sample)
            store_classes.append(sample_class)

    store_not_modified = True
    while len(handbag) > 0 and store_not_modified:
        store_not_modified = False
        indexes = []
        for index, s2 in enumerate(handbag):
            sample_class, sample = s2
            euc = []
            for s in store:
                euc.append(np.linalg.norm(s - sample))
            euc = np.array(euc)
            euc_nn = np.amin(euc)
            index_nn = np.ravel(np.where(euc == euc_nn))
            nn_class = store_classes[index_nn[0]]
            if nn_class != sample_class:
                store.append(sample)
                store_classes.append(sample_class)
                indexes.append(index)
                store_not_modified = True
        __delete_multiple_element__(handbag, indexes)
    del handbag

    store = np.array(store)
    X['data'] = store
    X['target'] = store_classes
    return X
