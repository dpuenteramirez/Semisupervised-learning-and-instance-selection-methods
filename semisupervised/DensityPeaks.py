#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    DensityPeaks.py
# @Author:      Daniel Puente Ramírez
# @Time:        5/3/22 09:55
# @Version:     3.1

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC


class STDPNF:

    def __init__(self,
                 dc=None,
                 distance_metric='euclidean',
                 k=3,
                 gauss_cutoff=True,
                 percent=2.0,
                 density_threshold=None,
                 distance_threshold=None,
                 anormal=True,
                 filtering=False,
                 classifier=None,
                 ):
        """Semi Supervised Algorithm based on Density Peaks."""
        self.dc = dc
        self.distance_metric = distance_metric
        self.k = k
        self.gauss_cutoff = gauss_cutoff
        self.percent = percent
        self.density_threshold = density_threshold
        self.distance_threshold = distance_threshold
        self.anormal = anormal
        self.filtering = filtering
        self.classifier = classifier

    def __build_distance(self):
        """
        Calculate distance dict.

        :return: distance dict, max distance, min distance
        """
        from scipy.spatial.distance import pdist, squareform

        distance_matrix = pdist(self.data, metric=self.distance_metric)
        distance_matrix = squareform(distance_matrix)

        triangle_upper = np.triu_indices(self.data.shape[0], 1)
        triangle_upper = distance_matrix[triangle_upper]

        distance = {}
        for i in range(self.n_id):
            for j in range(i + 1, self.n_id):
                distance[(i, j)] = distance_matrix[i, j]
                distance[(j, i)] = distance_matrix[i, j]

        max_dis, min_dis = np.max(triangle_upper), np.min(triangle_upper)

        return distance, max_dis, min_dis

    def __auto_select_dc(self):
        """
        Auto select the local density threshold that let average neighbor is 1-2 percent of all nodes.

        :return: dc that local density threshold
        """
        max_dis, min_dis = self.max_dis, self.min_dis
        dc = (max_dis + min_dis) / 2

        while True:
            nneighs = sum(
                [1 for v in self.distances.values() if v < dc]) / self.n_id ** 2
            if 0.01 <= nneighs <= 0.002:
                break
            # binary search
            if nneighs < 0.01:
                min_dis = dc
            else:
                max_dis = dc
            dc = (max_dis + min_dis) / 2
            if max_dis - min_dis < 0.0001:
                break
        return dc

    def __select_dc(self):
        """
        Select the local density threshold, default is the method used in paper, 'auto' is auto select.

        :return: dc that local density threshold
        """
        if self.dc == 'auto':
            dc = self.__auto_select_dc()
        else:
            position = int(self.n_id * (self.n_id + 1) / 2 * self.percent / 100)
            dc = np.sort(list(self.distances.values()))[
                position * 2 + self.n_id]

        return dc

    def __local_density(self):
        """
        Compute all points' local density.

        :return: local density vector that index is the point index
        """
        gauss_func = lambda dij, dc: math.exp(- (dij / dc) ** 2)
        cutoff_func = lambda dij, dc: 1 if dij < dc else 0
        func = gauss_func if self.gauss_cutoff else cutoff_func
        rho = [0] * self.n_id
        for i in range(self.n_id):
            for j in range(i + 1, self.n_id):
                temp = func(self.distances[(i, j)], self.dc)
                rho[i] += temp
                rho[j] += temp
        return np.array(rho, np.float32)

    def __min_neighbor_and_distance(self):
        """
        Compute all points' min util to the higher local density point(which is the nearest neighbor).

        :return: distance vector, nearest neighbor vector
        """
        sort_rho_idx = np.argsort(-self.rho)
        delta, nneigh = [float(self.max_dis)] * (self.n_id), [0] * self.n_id
        delta[sort_rho_idx[0]] = -1.
        for i in range(self.n_id):
            for j in range(0, i):
                old_i, old_j = sort_rho_idx[i], sort_rho_idx[j]
                if self.distances[(old_i, old_j)] < delta[old_i]:
                    delta[old_i] = self.distances[(old_i, old_j)]
                    nneigh[old_i] = old_j
        delta[sort_rho_idx[0]] = max(delta)

        return np.array(delta, np.float32), np.array(nneigh, np.float32)

    def __structure(self):
        self.structure = dict.fromkeys(range(self.n_id))
        for index, sample in enumerate(self.data):
            self.structure[index] = [
                sample,
                int(self.nneigh[index]),
                None,
                self.y[index] if index < len(self.y) else -1
            ]

        for index in range(self.n_id):
            if self.structure[self.structure[index][1]][2] is None:
                self.structure[self.structure[index][1]][2] = index

        self.structure = pd.DataFrame(self.structure, index=['sample', 'next',
                                                             'previous',
                                                             'label']) \
            .transpose()

        self.structure_stdnpf = self.structure.copy(deep=True)

    def __step_a(self):
        samples_labeled = self.structure.loc[self.structure['label'] != -1]
        l = samples_labeled['sample'].to_list()
        y_without = samples_labeled['label'].to_list()
        self.classifier.fit(l, y_without)
        return samples_labeled

    def __discover_structure(self):
        self._fit_without()

    def __nan_search(self):
        r = 1
        nan = defaultdict(set)
        nb = dict.fromkeys(range(self.n_id), 0)
        knn = defaultdict(set)
        rnn = defaultdict(set)
        cnt = defaultdict(int)

        while True:
            search = NearestNeighbors(n_neighbors=r + 1, algorithm='kd_tree')
            search.fit(self.data)
            for index, sample in enumerate(self.data):
                r_neighs = search.kneighbors([sample],
                                             return_distance=False)[0][1:]
                knn[index].update(list(r_neighs))
                for neigh in r_neighs:
                    nb[neigh] += 1
                    rnn[neigh].add(index)

            cnt[r] = np.count_nonzero((np.array(list(nb.values())) == 0))
            if r > 2 and cnt[r] == cnt[r - 1]:
                r -= 1
                break
            r += 1

        for index in range(self.n_id):
            nan[index] = knn[index].intersection(rnn[index])

        return nan, r

    def __enane(self, fx, nan, r):
        es = []
        es_pred = []
        local_structure = self.structure_stdnpf.copy(deep=True)
        base_estimator = KNeighborsClassifier(n_neighbors=r,
                                              metric=self.distance_metric)

        labeled_data = local_structure.loc[local_structure[
                                               'label'] != -1]
        nan_unlabeled = local_structure.loc[fx]
        data = pd.concat([labeled_data, nan_unlabeled], join='inner')

        enane_model = SelfTrainingClassifier(base_estimator)
        enane_model.fit(data['sample'].tolist(), data['label'].tolist())

        enane_pred = enane_model.predict(nan_unlabeled['sample'].tolist())

        for (row_index, _), pred in zip(nan_unlabeled.iterrows(),
                                        enane_pred):
            usefulness = 0
            harmfulness = 0
            for neigh in nan[row_index]:
                if local_structure.loc[neigh, 'label'] == pred:
                    usefulness += 1
                else:
                    harmfulness += 1

            if usefulness >= harmfulness:
                es.append(row_index)
                es_pred.append(pred)

        return es, es_pred

    def __init_values(self, l, u, y):
        self.y = y
        self.l = l
        self.u = u
        self.data = np.concatenate((l, u), axis=0)
        self.n_id = self.data.shape[0]
        self.distances, self.max_dis, self.min_dis = self.__build_distance()
        self.dc = self.__select_dc()
        self.rho = self.__local_density()
        self.delta, self.nneigh = self.__min_neighbor_and_distance()
        self.__structure()

    def _fit_without(self):
        if self.classifier is None:
            self.classifier = SVC()
        count = 1
        self.order = dict.fromkeys(range(self.n_id), 0)

        # Step 2
        while True:
            # 2.a
            samples_labeled = self.__step_a()

            # 2.b
            next_rows = samples_labeled['next'].to_numpy()
            next_unlabeled = []
            samples_labeled_index = samples_labeled.index.to_list()
            for next_row in next_rows:
                if next_row not in samples_labeled_index:
                    next_unlabeled.append(next_row)
                    self.order[next_row] = count
            if len(next_unlabeled) == 0:
                break
            unlabeled_next_of_labeled = self.structure.loc[next_unlabeled]

            lu = unlabeled_next_of_labeled['sample'].to_list()
            y_pred = self.classifier.predict(lu)

            # 2.c
            for new_label, pos in zip(y_pred, next_unlabeled):
                self.structure.at[pos, 'label'] = new_label

            # For STDPNF
            count += 1

        # Step 3
        while True:
            # 3.a
            samples_labeled = self.__step_a()

            # 3.b
            prev_rows = samples_labeled['previous'].to_numpy()
            prev_unlabeled = []
            samples_labeled_index = samples_labeled.index.to_list()
            for prev_row in prev_rows:
                if prev_row not in samples_labeled_index and prev_row is not \
                        None:
                    prev_unlabeled.append(prev_row)
                    self.order[prev_row] = count
            if len(prev_unlabeled) == 0:
                break
            unlabeled_prev_of_labeled = self.structure.loc[prev_unlabeled]

            lu = unlabeled_prev_of_labeled['sample'].to_list()
            y_pred = self.classifier.predict(lu)

            # 3.c
            for new_label, pos in zip(y_pred, prev_unlabeled):
                self.structure.at[pos, 'label'] = new_label

            # For STDPNF
            count += 1

    def _fit_stdpnf(self):
        """
        Self Training based on Density Peaks and a parameter-free noise
        filter.
        """

        self.__discover_structure()

        nan, lambda_param = self.__nan_search()
        self.classifier_stdpnf = KNeighborsClassifier(
            n_neighbors=self.k, metric=self.distance_metric)
        self.classifier_stdpnf.fit(self.l, self.y)
        count = 1

        while count <= max(self.order.values()):
            unlabeled_rows = self.structure_stdnpf.loc[self.structure_stdnpf[
                                                           'label'] == -1].\
                index.to_list()
            unlabeled_indexes = []
            for row in unlabeled_rows:
                if self.order[row] == count:
                    unlabeled_indexes.append(row)

            filtered_indexes, filtered_labels = self.__enane(
                unlabeled_indexes, nan, lambda_param)

            self.structure_stdnpf.at[filtered_indexes, 'label'] = \
                filtered_labels

            labeled_data = self.structure_stdnpf.loc[self.structure_stdnpf[
                                                         'label'] != -1]
            self.classifier_stdpnf.fit(
                labeled_data['sample'].tolist(), labeled_data['label'].tolist())

            count += 1

        labeled_data = self.structure_stdnpf.loc[self.structure_stdnpf[
                                                     'label'] != -1]
        self.classifier_stdpnf.fit(
            labeled_data['sample'].tolist(), labeled_data['label'].tolist())

    def fit(self, l, u, y):
        """Fit method."""
        if len(l) != len(y):
            raise ValueError(
                f'The dimension of the labeled data must be the same as the '
                f'number of labels given. {len(l)} != {len(y)}'
            )

        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        self.__init_values(l, u, y)
        if self.filtering:
            self._fit_stdpnf()
        else:
            self._fit_without()

    def predict(self, src):
        return self.classifier.predict(src)
