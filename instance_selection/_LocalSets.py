#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    _LocalSets.py
# @Author:      Daniel Puente Ramírez
# @Time:        18/3/22 11:14
# @Version:     2.0
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


class LocalSets:
    """
    Leyva, E., González, A., & Pérez, R. (2015). Three new instance selection
        methods based on local sets: A comparative study with several approaches
        from a bi-objective perspective. Pattern Recognition, 48(4), 1523-1537.
    """

    def __init__(self):
        self.local_sets = None
        self.n_id = 0

    def compute_local_sets(self, instances, labels):
        self.n_id = len(instances)
        structure = dict.fromkeys(range(self.n_id))
        distances = pairwise_distances(instances)

        for index, (sample, label) in enumerate(zip(instances, labels)):
            closest_enemy_distance = sys.maxsize
            closest_enemy_sample = None
            for index2, (_, label2) in enumerate(zip(instances, labels)):
                if index == index2 or label == label2:
                    continue
                if distances[index][index2] < closest_enemy_distance:
                    closest_enemy_distance = distances[index][index2]
                    closest_enemy_sample = index2
            structure[index] = [sample, [], None, closest_enemy_distance,
                                closest_enemy_sample, label]

        for index, (sample, label) in enumerate(zip(instances, labels)):
            neighs = []
            for index2, (_, label2) in enumerate(zip(instances, labels)):
                if index != index2 and label == label2 and \
                        distances[index][index2] < structure[index][3]:
                    neighs.append(index2)

            structure[index][1] = neighs
            structure[index][2] = len(neighs)

        self.local_sets = pd.DataFrame(structure, index=['sample',
                                                         'index_ls',
                                                         'LSC', 'LSR',
                                                         'enemy', 'label']) \
            .transpose()

    def sort_asc_lsc(self):
        self.local_sets = self.local_sets.sort_values(by='LSC')

    def usefulness(self, e):
        local_sets = self.local_sets
        local_sets = local_sets['index_ls'].values
        return len([x for x in local_sets if e in x])

    def get_local_sets(self):
        return self.local_sets

    @staticmethod
    def check_frame_to_numpy(y):
        if isinstance(y, pd.DataFrame):
            return np.ravel(y.to_numpy())
        return y


class LSSm(LocalSets):
    def __init__(self):
        super().__init__()

    def filter(self, instances, labels):
        names = instances.keys()
        instances = instances.to_numpy()
        instances = [np.ravel(i) for i in instances]
        labels = self.check_frame_to_numpy(labels)
        if len(instances) != len(labels):
            raise ValueError(
                f'The dimension of the labeled data must be the same as the '
                f'number of labels given. {len(instances)} != {len(labels)}'
            )
        self.n_id = len(instances)
        s_samples = []
        s_labels = []

        super(LSSm, self).compute_local_sets(instances, labels)
        for index in range(self.n_id):
            usefulness = super(LSSm, self).usefulness(index)
            try:
                harmfulness = super(LSSm, self).get_local_sets()['enemy']
                harmfulness = harmfulness.value_counts()[index]
            except KeyError:
                harmfulness = 0
            if usefulness >= harmfulness:
                s_samples.append(instances[index])
                s_labels.append(labels[index])

        x = pd.DataFrame(s_samples, columns=names)
        y = pd.DataFrame(s_labels)
        return x, y


class LSBo(LocalSets):
    def __init__(self):
        super(LSBo, self).__init__()

    def filter(self, instances, labels):
        names = instances.keys()
        if len(instances) != len(labels):
            raise ValueError(
                f'The dimension of the labeled data must be the same as the '
                f'number of labels given. {len(instances)} != {len(labels)}'
            )
        self.n_id = len(instances)
        labels = self.check_frame_to_numpy(labels)
        lssm = LSSm()
        instances, labels = lssm.filter(instances, labels)
        instances = instances.to_numpy()
        labels = labels.to_numpy().flatten()
        super(LSBo, self).compute_local_sets(instances, labels)
        super(LSBo, self).sort_asc_lsc()

        s_indexes = []
        for index, row in super(LSBo, self).get_local_sets().iterrows():
            is_in = False
            for ls_index in row['index_ls']:
                if ls_index in s_indexes:
                    is_in = True
                    break
            if not is_in:
                s_indexes.append(index)

        x = pd.DataFrame(instances[s_indexes], columns=names)
        y = pd.DataFrame(labels[s_indexes])

        return x, y
