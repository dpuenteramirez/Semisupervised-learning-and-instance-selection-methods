#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ICF.py
# @Author:      Daniel Puente Ramírez
# @Time:        23/11/21 09:37
# @Version:     3.0

from sys import maxsize

import numpy as np
import pandas as pd

from ._ENN import ENN
from .utils import transform, delete_multiple_element


class ICF:
    def __init__(self, nearest_neighbors=3, power_parameter=2):
        self.nearest_neighbors = nearest_neighbors
        self.power_parameter = power_parameter
        self.x_attr = None

    def __coverage__(self, train_set):
        """
        Inner method that performs the coverage and reachability of T.
        :param train_set: samples of T
        :return: (cov, reachable): vectors. cov contains for each sample the
        samples that it can cover. And reachable for each sample the samples
        that are able to reach it.
        """
        size = len(train_set.data)
        matrix_distances = np.zeros([size, size])
        distances_to_enemies = []
        sol = []

        for sample in range(size):
            distance_to_closest_enemy = maxsize
            x_sample = train_set['data'][sample]
            x_target = train_set['target'][sample]
            for other_sample in range(size):
                y_sample = train_set['data'][other_sample]
                y_target = train_set['target'][other_sample]
                distance = np.linalg.norm(x_sample - y_sample)
                matrix_distances[sample][other_sample] = distance
                # Si son enemigas, nos quedamos con la distancia
                if x_target != y_target and \
                        distance < distance_to_closest_enemy:
                    distance_to_closest_enemy = distance
            distances_to_enemies.append(distance_to_closest_enemy)

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

        for sample in range(size):
            reachable = []
            x_target = train_set['target'][sample]
            for other_sample in range(size):
                y_target = train_set['target'][other_sample]
                if sample != other_sample and x_target == y_target:
                    # if x_target == y_target:
                    coverage = sol[other_sample][0]
                    if sample in coverage:
                        reachable.append(other_sample)
            sol[sample].append(reachable)

        return sol

    def filter(self, samples, y):
        """
        Implementation of Iterative Case Filtering

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

            cov_reach = self.__coverage__(ts)

            progress = False
            removable_indexes = []
            for index in range(len(cov_reach)):
                (x_cov, x_reach) = cov_reach[index]
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
