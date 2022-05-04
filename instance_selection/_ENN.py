#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ENN.py
# @Author:      Daniel Puente Ramírez
# @Time:        16/11/21 17:14
# @Version:     7.0

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .utils import transform, transform_original_complete


class ENN:

    """
    Wilson, D. L. (1972). Asymptotic properties of nearest neighbor rules
    using edited data. IEEE Transactions on Systems, Man, and
    Cybernetics, (3), 408-421.

    Parameters
    ----------
    nearest_neighbors : int, default=3
        Number to use as nearest neighbors when computing distances.

    power_parameter : int, default=2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance (l2)
        for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    """

    def __init__(self, nearest_neighbors=3, power_parameter=2):
        """
        The function takes in two parameters, nearest_neighbors
        and power_parameter, and assigns them to the attributes
        nearest_neighbors and power_parameter

        :param nearest_neighbors: The number of nearest neighbors to use when
        calculating the weights, defaults to 3 (optional)
        :param power_parameter: This is the exponent that is used to calculate
        the weights, defaults to 2 (optional)
        """
        self.nearest_neighbors = nearest_neighbors
        self.power_parameter = power_parameter
        self.x_attr = None

    def _neighs(self, s_samples, s_targets, index, removed):
        """
        _neighs() takes in the samples and targets, the index of the sample to
        be removed, and the number of samples already removed. It returns the
        sample to be removed, its target, the targets of the samples not yet
        removed, the samples not yet removed, and the indices of the nearest
        neighbors of the sample to be removed.

        :param s_samples: the samples that are being used to train the model
        :param s_targets: the targets of the samples
        :param index: the index of the sample to be removed
        :param removed: the number of samples that have been removed from the
        dataset
        """
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
            _, x_target, targets_not_x, samples_not_x, neigh_ind = \
                self._neighs(s_samples, s_targets, index, removed)
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
                self._neighs(s_samples, s_targets, index, removed)
            y_targets = [targets_not_x[x] for x in neigh_ind[0]]
            count = np.bincount(np.ravel(y_targets))
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
