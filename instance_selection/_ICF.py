#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ICF.py
# @Author:      Daniel Puente Ramírez
# @Time:        23/11/21 09:37
# @Version:     4.0

from sys import maxsize

import numpy as np
import pandas as pd

from ._ENN import ENN
from .utils import transform, delete_multiple_element


class ICF:
    """
    Brighton, H., & Mellish, C. (2002). Advances in instance selection for
    instance-based learning algorithms. Data mining and knowledge
    discovery, 6(2), 153-172.

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
        self.nearest_neighbors = nearest_neighbors
        self.power_parameter = power_parameter
        self.x_attr = None

    def filter(self, samples, y):
        """
        Implementation of Iterative Case Filtering.

        ICF is based on coverage and reachable, due to this two concepts it
        performs deletion of samples based on the rule: "If the reachability
        of a sample is greater than its coverage, that sample has to be
        removed".
        I.e. if another sample of the same class which is nearer than the
        closest enemy is able to provide more information, the sample with
        less coverage than reachable will be removed.
        Ending up with a dataset as generalized as ICF allows.

        :param samples: DataFrame.
        :param y: DataFrame.
        :return: the input dataset with the remaining samples.
        """
        self.x_attr = samples.keys()
        enn = ENN(nearest_neighbors=self.nearest_neighbors,
                  power_parameter=self.power_parameter)
        samples, y = enn.filter(samples, y)
        ts = transform(samples, y)

        while True:
            data = list(ts['data'])
            target = list(ts['target'])

            cov_reach = self._coverage(ts)

            progress = False
            removable_indexes = []
            for index, (x_cov, x_reach) in enumerate(cov_reach):
                if len(x_reach) > len(x_cov):
                    removable_indexes.append(index)
                    progress = True

            delete_multiple_element(data, removable_indexes)
            delete_multiple_element(target, removable_indexes)
            ts['data'] = data
            ts['target'] = target
            if not progress:
                break

        samples = pd.DataFrame(ts['data'], columns=self.x_attr)
        y = pd.DataFrame(ts['target'])

        return samples, y

    @staticmethod
    def _coverage(train_set):
        """
        For each sample, it finds the closest enemy and the samples that can
        reach it.

        :param train_set: the training set
        :return: A list of lists. Each list contains two lists. The first one is
         the list of samples that can be covered by the sample. The second
         one is the
        list of samples that can reach the sample.
        """
        size = len(train_set.data)
        matrix_distances = np.zeros([size, size])
        distances_to_enemies = []
        sol = []

        ICF.calculate_distances_to_enemies(distances_to_enemies,
                                           matrix_distances,
                                           size, train_set)

        ICF.calculate_coverage(distances_to_enemies, matrix_distances, size,
                               sol, train_set)

        ICF.calculate_reachability(size, sol, train_set)

        return sol

    @staticmethod
    def calculate_reachability(size, sol, train_set):
        """
        For each sample, find all other samples that are reachable from it

        :param size: the number of samples in the training set
        :param sol: the solution matrix
        :param train_set: the training set
        """
        for sample in range(size):
            reachable = []
            x_target = train_set['target'][sample]
            for other_sample in range(size):
                y_target = train_set['target'][other_sample]
                if sample != other_sample and x_target == y_target:
                    coverage = sol[other_sample][0]
                    if sample in coverage:
                        reachable.append(other_sample)
            sol[sample].append(reachable)

    @staticmethod
    def calculate_coverage(distances_to_enemies, matrix_distances, size, sol,
                           train_set):
        """
        For each sample, we find the distance to the closest enemy. Then, we
        find all the samples that are closer to the current sample than the
        closest enemy

        :param distances_to_enemies: a list of distances to the closest enemy
        for each sample
        :param matrix_distances: a matrix of distances between all samples in
        the training set
        :param size: the number of samples in the training set
        :param sol: the solution matrix
        :param train_set: the training set
        """
        for sample in range(size):
            x_coverage = []
            x_target = train_set['target'][sample]
            distance_to_closest_enemy = distances_to_enemies[sample]
            for other_sample in range(size):
                if sample == other_sample:
                    continue
                y_target = train_set['target'][other_sample]
                if x_target == y_target:
                    distance_between_samples = matrix_distances[sample][
                        other_sample]
                    if distance_between_samples < distance_to_closest_enemy:
                        x_coverage.append(other_sample)
            sol.append([x_coverage])

    @staticmethod
    def calculate_distances_to_enemies(distances_to_enemies,
                                       matrix_distances, size, train_set):
        """
        For each sample in the training set, calculate the distance to the
        closest enemy and store it in the distances_to_enemies array

        :param distances_to_enemies: a list of distances to the closest enemy
        for each sample
        :param matrix_distances: a matrix of size (size,size) where size is the
        number of samples in the train set
        :param size: the number of samples in the training set
        :param train_set: the training set
        """
        for sample in range(size):
            distance_to_closest_enemy = maxsize
            x_sample = train_set['data'][sample]
            x_target = train_set['target'][sample]
            for other_sample in range(size):
                y_sample = train_set['data'][other_sample]
                y_target = train_set['target'][other_sample]
                distance = np.linalg.norm(x_sample - y_sample)
                matrix_distances[sample][other_sample] = distance

                if x_target != y_target and \
                        distance < distance_to_closest_enemy:
                    distance_to_closest_enemy = distance
            distances_to_enemies.append(distance_to_closest_enemy)
