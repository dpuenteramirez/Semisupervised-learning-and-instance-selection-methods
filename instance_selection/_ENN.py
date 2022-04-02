#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ENN.py
# @Author:      Daniel Puente Ramírez
# @Time:        16/11/21 17:14
# @Version:     6.0

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .utils import transform, transform_original_complete


class ENN:
    def __init__(self, nearest_neighbors=3, power_parameter=2):
        self.nearest_neighbors = nearest_neighbors
        self.power_parameter = power_parameter
        self.x_attr = None

    def neighs(self, s_samples, s_targets, index, removed):
        x_sample = s_samples[index - removed]
        x_target = s_targets[index - removed]
        knn = NearestNeighbors(n_jobs=-1,
                               n_neighbors=self.nearest_neighbors, p=2)
        samples_not_x = s_samples[:index - removed] + s_samples[
                                                      index - removed + 1:]
        targets_not_x = s_targets[:index - removed] + s_targets[
                                                      index - removed + 1:]
        knn.fit(samples_not_x)
        _, neigh_ind = knn.kneighbors([x_sample])

        return x_sample, x_target, targets_not_x, samples_not_x, neigh_ind

    def filter(self, samples, y):
        """
        Wilson, D. L. (1972). Asymptotic properties of nearest neighbor rules
            using edited data. IEEE Transactions on Systems, Man, and
            Cybernetics, (3), 408-421.

        Implementation of the Wilson Editing algorithm.

        For each sample locates the *k* nearest neighbors and selects the
        number of different classes there are.
        If a sample results in a wrong classification after being classified
        with k-NN, that sample is removed from the TS.
        :param samples: DataFrame.
        :param y: DataFrame.
        :return: the input dataset with the remaining samples.
        """
        self.x_attr = samples.keys()
        samples = transform(samples, y)
        size = len(samples['data'])
        s_samples = list(samples['data'])
        s_targets = list(samples['target'])
        removed = 0

        for index in range(size):
            _, x_target, targets_not_x, samples_not_x, neigh_ind = self.neighs(
                s_samples, s_targets, index, removed)
            y_targets = np.ravel(
                np.array([targets_not_x[x] for x in neigh_ind[0]])).astype(int)
            count = np.bincount(y_targets)
            max_class = np.where(count == np.amax(count))[0][0]
            if max_class != x_target:
                removed += 1
                s_samples = samples_not_x
                s_targets = targets_not_x

        samples = pd.DataFrame(s_samples, columns=self.x_attr)
        y = pd.DataFrame(s_targets)

        return samples, y

    def filter_original_complete(self, original, original_y, complete,
                                 complete_y):
        """
        Modification of the Wilson Editing algorithm.

        For each sample locates the *k* nearest neighbors and selects the number
        of different classes there are.
        If a sample results in a wrong classification after being classified
        with k-NN, that sample is removed from the TS, only if the sample to be
        removed is not from the original dataset.
        :param original: DataFrame: dataset with the initial samples.
        :param original_y: DataFrame: labels.
        :param complete: DataFrame: dataset with the initial samples and the new
        ones added by self-training.
        :param complete_y: labels.
        :return: the input dataset with the remaining samples.
        """
        self.x_attr = original.keys()
        original, complete = transform_original_complete(original, original_y,
                                                         complete, complete_y)
        size = len(complete['data'])
        s_samples = list(complete['data'])
        s_targets = list(complete['target'])
        o_samples = list(original['data'])
        removed = 0

        for index in range(size):
            x_sample, x_target, targets_not_x, samples_not_x, neigh_ind = \
                self.neighs(s_samples, s_targets, index, removed)
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

        samples = pd.DataFrame(s_samples, columns=self.x_attr)
        y = pd.DataFrame(s_targets)

        return samples, y
