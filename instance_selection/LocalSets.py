#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    LocalSets.py
# @Author:      Daniel Puente Ramírez
# @Time:        18/3/22 11:14
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances


class LSSm:
    def __init__(self):
        self.__local_sets = None
        self.n_id = 0

    def __compute_local_sets(self, instances, labels):
        structure = dict.fromkeys(range(self.n_id))
        distances = pairwise_distances(instances)

        for index, (sample, label) in enumerate(zip(instances, labels)):
            closest_enemy_distance = sys.maxsize
            closest_enemy_sample = None
            for index2, (sample2, label2) in enumerate(zip(instances, labels)):
                if index == index2 or label == label2:
                    continue
                if distances[index][index2] < closest_enemy_distance:
                    closest_enemy_distance = distances[index][index2]
                    closest_enemy_sample = index2
            structure[index] = [sample, [], None, closest_enemy_distance,
                                closest_enemy_sample]

        for index, (sample, label) in enumerate(zip(instances, labels)):
            neighs = []
            for index2, (sample2, label2) in enumerate(zip(instances, labels)):
                if index != index2 and label == label2 and \
                        distances[index][index2] < structure[index][3]:
                    neighs.append(index2)

            structure[index][1] = neighs
            structure[index][2] = len(neighs)

        self.__local_sets = pd.DataFrame(structure, index=['sample',
                                                           'index_ls',
                                                           'LSC', 'LSR',
                                                           'enemy']) \
            .transpose()

    def __usefulness(self, e):
        local_sets = self.__local_sets['index_ls'].values
        return len([x for x in local_sets if e in x])

    def filter(self, instances, labels):
        names = instances.keys()
        instances = instances.to_numpy()
        if len(instances) != len(labels):
            raise ValueError(
                f'The dimension of the labeled data must be the same as the '
                f'number of labels given. {len(instances)} != {len(labels)}'
            )
        self.n_id = len(instances)
        s_samples = []
        s_labels = []

        self.__compute_local_sets(instances, labels)
        for index in range(self.n_id):
            usefulness = self.__usefulness(index)
            try:
                harmfulness = self.__local_sets['enemy'].value_counts()[index]
            except KeyError:
                harmfulness = 0
            if usefulness >= harmfulness:
                s_samples.append(instances[index])
                s_labels.append(labels[index])

        X = pd.DataFrame(s_samples, columns=names)
        y = pd.DataFrame(s_labels)
        return X, y


if __name__ == '__main__':
    iris = load_iris()
    print(iris.keys())
    X1 = pd.DataFrame(iris.data, columns=iris.feature_names)
    y1 = iris.target

    model = LSSm()
    X2, y2 = model.filter(X1, y1)
    print(len(X2))
