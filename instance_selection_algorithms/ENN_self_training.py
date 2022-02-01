#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ENN_self_training.py
# @Author:      Daniel Puente Ramírez
# @Time:        26/1/22 11:24
import copy

import numpy as np
from sklearn.neighbors import NearestNeighbors

from instance_selection_algorithms.ENN import ENN


def ENN_self_training(original, complete, k=3, without=True):
    """
    Modification of the Wilson Editing algorithm.

    For each sample locates the *k* nearest neighbors and selects the number
    of different classes there are.
    If a sample results in a wrong classification after being classified
    with k-NN, that sample is removed from the TS, only if the sample to be
    removed is not from the original dataset.
    :param original: Bunch: dataset with the initial samples.
    :param complete: Bunch: dataset with the initial samples and the new
    ones added by self-training.
    :param k: int: number of neighbors to evaluate.
    :param without: Boolean: True if it must use the modified ENN (
    without deleting the original samples), False if it must perform ENN
    by Wilson guidelines.
    :return: the input dataset with the remaining samples.
    """

    if not without:
        return ENN(complete, k)

    S = copy.deepcopy(complete)
    size = len(complete['data'])
    s_samples = list(complete['data'])
    s_targets = list(complete['target'])
    o_samples = list(original['data'])
    removed = 0

    for index in range(size):
        x_sample = s_samples[index - removed]
        x_target = s_targets[index - removed]
        knn = NearestNeighbors(n_jobs=-1, n_neighbors=k, p=2)
        samples_not_x = s_samples[:index - removed] + s_samples[
                                                      index - removed + 1:]
        targets_not_x = s_targets[:index - removed] + s_targets[
                                                      index - removed + 1:]
        knn.fit(samples_not_x)
        _, neigh_ind = knn.kneighbors([x_sample])
        y_targets = [targets_not_x[x] for x in neigh_ind[0]]
        count = np.bincount(y_targets)
        max_class = np.where(count == np.amax(count))[0][0]
        if max_class != x_target:
            delete = True
            for o_sample in o_samples:
                if np.array_equal(o_sample, x_sample):
                    delete = False
            if delete:
                removed += 1
                s_samples = samples_not_x
                s_targets = targets_not_x

    S['data'] = np.array(s_samples)
    S['target'] = s_targets

    return S
