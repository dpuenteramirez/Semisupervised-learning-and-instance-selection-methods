#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    densitypeaks.py
# @Author:      Daniel Puente Ramírez
# @Time:        5/3/22 09:55

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import math


class SemiSupervisedSelfTraining:

    def __init__(self,
                 dc=None,
                 distance_metric='euclidean',
                 gauss_cutoff=True,
                 density_threshold=None,
                 distance_threshold=None,
                 anormal=True,
                 filtering=False
                 ):
        self.dc = dc
        self.distance_metric = distance_metric
        self.gauss_cutoff = gauss_cutoff
        self.density_threshold = density_threshold
        self.distance_threshold = distance_threshold
        self.anormal = anormal
        self.filtering = filtering

    def build_distance(self):
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

    def auto_select_dc(self):
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

    def select_dc(self):
        """
        Select the local density threshold, default is the method used in paper, 'auto' is auto select.

        :return: dc that local density threshold
        """
        if self.dc == 'auto':
            dc = self.auto_select_dc()
        else:
            percent = 2.0
            position = int(self.n_id * (self.n_id + 1) / 2 * percent / 100)
            dc = np.sort(list(self.distances.values()))[
                position * 2 + self.n_id]

        return dc

    def local_density(self):
        """
        Compute all points' local density.

        :return: local density vector that index is the point index
        """
        guass_func = lambda dij, dc: math.exp(- (dij / dc) ** 2)
        cutoff_func = lambda dij, dc: 1 if dij < dc else 0
        func = guass_func if self.gauss_cutoff else cutoff_func
        rho = [0] * self.n_id
        for i in range(self.n_id):
            for j in range(i + 1, self.n_id):
                temp = func(self.distances[(i, j)], self.dc)
                rho[i] += temp
                rho[j] += temp
        return np.array(rho, np.float32)

    def min_neighbor_and_distance(self):
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

    def _fit_without(self, L, U, y):
        self.data = np.concatenate((L, U), axis=0)
        self.n_id = self.data.shape[0]
        self.distances, self.max_dis, self.min_dis = self.build_distance()
        self.dc = self.select_dc()
        self.rho = self.local_density()
        self.delta, self.nneigh = self.min_neighbor_and_distance()
        self.dlabeled = {}
        self.dunlabeled = {}
        self.classifier = SVC()

        for index, sample in enumerate(L):
            self.dlabeled[tuple(sample)] = tuple(self.data[int(self.nneigh[
                                                                   index])])
            try:
                self.dlabeled[tuple(sample)]
            except KeyError:
                print(tuple(sample))
                print(tuple(self.data[int(self.nneigh[index])]))
                self.dlabeled[tuple(sample)] = tuple(self.data[int(
                    self.nneigh[index])])

        while len(self.dlabeled.keys()) != len(y):
            y = y[:-1]

        for index, sample in enumerate(U):
            self.dunlabeled[tuple(sample)] = \
                tuple(self.data[int(self.nneigh[index + len(L)])])
            try:
                self.dunlabeled[tuple(sample)]
            except KeyError:
                print(tuple(sample))
                print(tuple(self.data[int(self.nneigh[index])]))
                self.dunlabeled[tuple(sample)] = tuple(self.data[int(
                    self.nneigh[index + len(L)])])

        while True:
            L = list(self.dlabeled.keys())
            self.classifier.fit(L, y)
            samples_to_add = []
            for sample, next in self.dunlabeled.items():
                if next in self.dlabeled.keys():
                    samples_to_add.append(tuple(sample))
            if len(samples_to_add) == 0:
                break
            next_pred = self.classifier.predict(samples_to_add)
            y = np.concatenate((next_pred, y), axis=0)

            for sample in samples_to_add:
                self.dlabeled[tuple(sample)] = self.dunlabeled[tuple(sample)]
                self.dunlabeled.pop(tuple(sample))
            while len(self.dlabeled.keys()) != len(y):
                y = y[:-1]

        while len(self.dunlabeled) > 0:
            L = list(self.dlabeled.keys())
            self.classifier.fit(L, y)
            samples_to_add = []
            for prev, sample in self.dlabeled.items():
                if tuple(sample) in self.dunlabeled.keys():
                    samples_to_add.append(sample)
            if len(samples_to_add) == 0:
                break
            next_pred = self.classifier.predict(samples_to_add)
            y = np.concatenate((next_pred, y), axis=0)
            for sample in samples_to_add:
                self.dlabeled[tuple(sample)] = self.dunlabeled[tuple(sample)]
                self.dunlabeled.pop(tuple(sample))
            while len(self.dlabeled.keys()) != len(y):
                y = y[:-1]

    def fit(self, L, U, y):
        """Fit method"""
        if len(L) != len(y):
            raise ValueError(
                f'The dimension of the labeled data must be the same as the '
                f'number of labels given. {len(L)} != {len(y)}'
            )

        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        if self.filtering is False:
            self._fit_without(L, U, y)

    def predict(self, src):
        return self.classifier.predict(src)


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
    choices = np.random.choice(len(X_train), math.ceil(len(X_train) * 0.5),
                               replace=False)
    labeled = [x for x in range(len(X_train)) if x not in choices]
    y = y[labeled]
    X_labeled = X_train[labeled]
    X_unlabeled = X_train[choices]

    model = SemiSupervisedSelfTraining(filtering=False)
    model.fit(X_labeled, X_unlabeled, y)
    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_pred))


