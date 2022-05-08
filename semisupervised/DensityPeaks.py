#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    DensityPeaks.py
# @Author:      Daniel Puente Ramírez
# @Time:        5/3/22 09:55
# @Version:     4.0

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC

from instance_selection import ENN

from .utils import split


class STDPNF:
    """
    Li, J., Zhu, Q., & Wu, Q. (2019). A self-training method based on density
    peaks and an extended parameter-free local noise filter for k nearest
    neighbor. Knowledge-Based Systems, 184, 104895.

    Wu, D., Shang, M., Luo, X., Xu, J., Yan, H., Deng, W., & Wang, G. (2018).
    Self-training semi-supervised classification based on density peaks of
    data. Neurocomputing, 275, 180-191.
    """

    def __init__(
        self,
        dc=None,
        distance_metric="euclidean",
        k=3,
        gauss_cutoff=True,
        percent=2.0,
        density_threshold=None,
        distance_threshold=None,
        anormal=True,
        filtering=False,
        classifier=None,
        classifier_params=None,
        filter_method=None,
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
        if classifier is not None:
            if isinstance(classifier_params, dict):
                self.classifier = classifier(**classifier_params)
            else:
                self.classifier = classifier()
        else:
            self.classifier = None
        if filter_method is not None and filter_method != "ENANE":
            self.filter = filter_method()
        elif isinstance(filter_method, str) and filter_method == "ENANE":
            self.filter = filter_method
        else:
            self.filter = None

        self.y = None
        self.low = None
        self.u = None
        self.classifier_stdpnf = None
        self.order = None
        self.structure = None
        self.structure_stdnpf = None
        self.n_id = None
        self.distances = None
        self.max_dis = None
        self.min_dis = None
        self.rho = None
        self.delta = None
        self.nneigh = None
        self.data = None

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
        Auto select the local density threshold that let average neighbor is 1-2
         percent of all nodes.

        :return: dc that local density threshold
        """
        max_dis, min_dis = self.max_dis, self.min_dis
        dc = (max_dis + min_dis) / 2

        while True:
            nneighs = (
                sum([1 for v in self.distances.values() if v < dc]) / self.n_id**2
            )
            if 0.01 <= nneighs <= 0.02:
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
        Select the local density threshold, default is the method used in paper,
        'auto' is auto select.

        :return: dc that local density threshold
        """
        if self.dc == "auto":
            dc = self.__auto_select_dc()
        else:
            position = int(self.n_id * (self.n_id + 1) /
                           2 * self.percent / 100)
            dc = np.sort(list(self.distances.values()))[
                position * 2 + self.n_id]

        return dc

    def __local_density(self):
        """
        Compute all points' local density.

        :return: local density vector that index is the point index
        """

        def gauss_func(dij, dc):
            """
            > The function takes in a distance value and a cutoff value, and
            returns the value of the Gaussian function at that point

            :param dij: distance between two nodes
            :param dc: The cutoff distance
            :return: the value of the gaussian function.
            """
            return math.exp(-((dij / dc) ** 2))

        def cutoff_func(dij, dc):
            """
            If the distance between two atoms is less than the cutoff distance,
            return 1, otherwise return 0

            :param dij: distance between atoms i and j
            :param dc: cutoff distance
            :return: 1 if dij < dc, else 0
            """
            return 1 if dij < dc else 0

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
        Compute all points' min util to the higher local density point(which is
         the nearest neighbor).

        :return: distance vector, nearest neighbor vector
        """
        if self.rho is None:
            raise ValueError("Encountered rho as None.")

        sort_rho_idx = np.argsort(-self.rho)
        delta, nneigh = [float(self.max_dis)] * self.n_id, [0] * self.n_id
        delta[sort_rho_idx[0]] = -1.0

        for i in range(self.n_id):
            for j in range(0, i):
                old_i, old_j = sort_rho_idx[i], sort_rho_idx[j]
                if self.distances[(old_i, old_j)] < delta[old_i]:
                    delta[old_i] = self.distances[(old_i, old_j)]
                    nneigh[old_i] = old_j
        delta[sort_rho_idx[0]] = max(delta)

        return np.array(delta, np.float32), np.array(nneigh, np.float32)

    def __structure(self):
        """
        The function takes the data and the nearest neighbor indices and creates
        a dataframe with the following columns:

        - sample: the data point
        - next: the index of the nearest neighbor
        - previous: the index of the nearest neighbor of the nearest neighbor
        - label: the label of the data point

        The function also creates a copy of the dataframe called
        structure_stdnpf
        """
        self.structure = dict.fromkeys(range(self.n_id))
        for index, sample in enumerate(self.data):
            self.structure[index] = [
                sample,
                int(self.nneigh[index]),
                None,
                self.y[index] if index < len(self.y) else -1,
            ]

        for index in range(self.n_id):
            if self.structure[self.structure[index][1]][2] is None:
                self.structure[self.structure[index][1]][2] = index

        self.structure = pd.DataFrame(
            self.structure, index=["sample", "next", "previous", "label"]
        ).transpose()

        self.structure_stdnpf = self.structure.copy(deep=True)

    def __step_a(self):
        """
        > The function takes the labeled samples and trains the classifier on
        them
        :return: The samples that have been labeled.
        """
        samples_labeled = self.structure.loc[self.structure["label"] != -1]
        sam_lab = samples_labeled["sample"].to_list()
        y_without = samples_labeled["label"].to_list()
        self.classifier.fit(sam_lab, y_without)
        return samples_labeled

    def __discover_structure(self):
        self._fit_without()

    def __nan_search(self):
        """
        For each point, find the set of points that are within a distance of r,
        and the set of points that are within a distance of r+1.

        The set of points that are within a distance of r+1 is a superset of the
        set of points that are within a distance of r.

        The set of points that are within a distance of r+1 is also a superset
        of the set of points that are within a distance of r+2.

        The set of points that are within a distance of r+2 is also a superset
        of the set of points that are within a distance of r+3.

        And so on.

        The set of points that are within a distance of r+1 is also a superset
        of the set of points that are within a distance of r+2.

        The set of points that are within a distance of r+2 is
        :return: nan, r
        """
        r = 1
        nan = defaultdict(set)
        nb = dict.fromkeys(range(self.n_id), 0)
        knn = defaultdict(set)
        rnn = defaultdict(set)
        cnt = defaultdict(int)

        while True:
            search = NearestNeighbors(n_neighbors=r + 1, algorithm="kd_tree")
            search.fit(self.data)
            for index, sample in enumerate(self.data):
                r_neighs = search.kneighbors(
                    [sample], return_distance=False)[0][1:]
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
        """
        > The function takes in the dataframe, the list of indices of the
        unlabeled data, the list of indices of the neighbors of the unlabeled
        data, and the number of neighbors to use in the KNN classifier. It
        then creates a new dataframe with the labeled data and the unlabeled
        data, and uses the KNN classifier to predict the labels of the
        unlabeled data. It then checks if the predicted label is the same as
        the label of the majority of the neighbors of the unlabeled data. If
        it is, then it adds the index of the unlabeled data to the list of
        indices of the data to be labeled

        :param fx: the indexes of the unlabeled data
        :param nan: a list of lists, where each list contains the indices of the
        neighbors of a sample
        :param r: the number of neighbors to consider
        :return: The indexes of the samples that are going to be labeled and the
        labels that are going to be assigned to them.
        """
        es = []
        es_pred = []
        local_structure = self.structure_stdnpf.copy(deep=True)
        base_estimator = KNeighborsClassifier(
            n_neighbors=r, metric=self.distance_metric
        )

        labeled_data = local_structure.loc[local_structure["label"] != -1]
        nan_unlabeled = local_structure.loc[fx]
        data = pd.concat([labeled_data, nan_unlabeled], join="inner")

        enane_model = SelfTrainingClassifier(base_estimator)
        enane_model.fit(data["sample"].tolist(), data["label"].tolist())

        enane_pred = enane_model.predict(nan_unlabeled["sample"].tolist())

        for (row_index, _), pred in zip(nan_unlabeled.iterrows(), enane_pred):
            usefulness = 0
            harmfulness = 0
            for neigh in nan[row_index]:
                if local_structure.loc[neigh, "label"] == pred:
                    usefulness += 1
                else:
                    harmfulness += 1

            if usefulness >= harmfulness:
                es.append(row_index)
                es_pred.append(pred)

        return es, es_pred

    def __init_values(self, low, u, y):
        """
        It takes in the lower and upper bounds of the data, and the data itself,
         and then calculates the distances between the data points,
         the maximum distance, the minimum distance, the dc value, the rho
         value, the delta value, the number of neighbors, and the structure
         of the data

        :param low: lower bound of the data
        :param u: upper bound of the data
        :param y: the labels of the data
        """
        self.y = y
        self.low = low
        self.u = u
        self.data = np.concatenate((low, u), axis=0)
        self.n_id = self.data.shape[0]
        self.distances, self.max_dis, self.min_dis = self.__build_distance()
        self.dc = self.__select_dc()
        self.rho = self.__local_density()
        self.delta, self.nneigh = self.__min_neighbor_and_distance()
        self.__structure()

    def _fit_without(self):
        """
        The function takes in a classifier, and then labels the next point,
        and then labels the previous points, without filtering.
        """
        if self.classifier is None:
            self.classifier = SVC()
        count = 1
        self.order = dict.fromkeys(range(self.n_id), 0)

        count = self._label_next_point(count)

        self._label_previous_points(count)

    def _label_previous_points(self, count):
        """
        > The function takes the samples labeled in the previous step and finds
         the previous samples of those samples. It then labels those samples
         and repeats the process until there are no more samples to label

        :param count: the number of the current iteration
        """
        while True:
            samples_labeled = self.__step_a()

            prev_rows = samples_labeled["previous"].to_numpy()
            prev_unlabeled = []
            samples_labeled_index = samples_labeled.index.to_list()
            for prev_row in prev_rows:
                if prev_row not in samples_labeled_index and prev_row is not None:
                    prev_unlabeled.append(prev_row)
                    self.order[prev_row] = count
            if len(prev_unlabeled) == 0:
                break
            unlabeled_prev_of_labeled = self.structure.loc[prev_unlabeled]

            lu = unlabeled_prev_of_labeled["sample"].to_list()
            y_pred = self.classifier.predict(lu)

            for new_label, pos in zip(y_pred, prev_unlabeled):
                self.structure.at[pos, "label"] = new_label

            count += 1

    def _label_next_point(self, count):
        """
        > The function takes the samples labeled in the previous step and finds
         the next samples in the structure. If the next samples are not
         labeled, it labels them and updates the order of the samples

        :param count: the number of the next point to be labeled
        :return: The number of labeled samples.
        """
        while True:
            samples_labeled = self.__step_a()

            next_rows = samples_labeled["next"].to_numpy()
            next_unlabeled = []
            samples_labeled_index = samples_labeled.index.to_list()
            for next_row in next_rows:
                if next_row not in samples_labeled_index:
                    next_unlabeled.append(next_row)
                    self.order[next_row] = count
            if len(next_unlabeled) == 0:
                break
            unlabeled_next_of_labeled = self.structure.loc[next_unlabeled]

            lu = unlabeled_next_of_labeled["sample"].to_list()
            y_pred = self.classifier.predict(lu)

            for new_label, pos in zip(y_pred, next_unlabeled):
                self.structure.at[pos, "label"] = new_label

            count += 1
        return count

    def _fit_stdpnf(self):
        """
        Self Training based on Density Peaks and a parameter-free noise
        filter.
        """

        self.__discover_structure()

        nan, lambda_param = self.__nan_search()
        self.classifier_stdpnf = KNeighborsClassifier(
            n_neighbors=self.k, metric=self.distance_metric
        )
        self.classifier_stdpnf.fit(self.low, self.y)
        count = 1

        while count <= max(self.order.values()):
            unlabeled_rows = self.structure_stdnpf.loc[
                self.structure_stdnpf["label"] == -1
            ].index.to_list()
            unlabeled_indexes = []
            for row in unlabeled_rows:
                if self.order[row] == count:
                    unlabeled_indexes.append(row)

            if isinstance(self.filter, str) and self.filter == "ENANE":
                filtered_indexes, filtered_labels = self.__enane(
                    unlabeled_indexes, nan, lambda_param
                )
                self.structure_stdnpf.at[filtered_indexes,
                                         "label"] = filtered_labels

            else:
                labeled_data = self.structure_stdnpf.loc[
                    self.structure_stdnpf["label"] != -1
                ]
                complete = labeled_data["sample"]
                complete_y = labeled_data["label"]

                result = self._if_filter(complete, complete_y)

                self._results_to_structure(complete, result)

            labeled_data = self.structure_stdnpf.loc[
                self.structure_stdnpf["label"] != -1
            ]
            self.classifier_stdpnf.fit(
                labeled_data["sample"].tolist(), labeled_data["label"].tolist()
            )

            count += 1

        labeled_data = self.structure_stdnpf.loc[self.structure_stdnpf["label"] != -1]
        self.classifier_stdpnf.fit(
            labeled_data["sample"].tolist(), labeled_data["label"].tolist()
        )

    def _results_to_structure(self, complete, result):
        """
        > This function takes the results of the model and compares them to the
        complete data set. If the result is not in the complete data set, it is
        added to the structure data set.

        :param complete: the complete dataset
        :param result: the result of the clustering
        """
        results_to_unlabeled = []
        for r in result.to_numpy():
            is_in = False
            for c in complete:
                if np.array_equal(r, c):
                    is_in = True
            if not is_in:
                results_to_unlabeled.append(r)
        for r in results_to_unlabeled:
            self.structure_stdnpf.at[np.array(self.structure_stdnpf["sample"], r)][
                "label"
            ] = -1

    def _if_filter(self, complete, complete_y):
        """
        If the filter is an ENN, then filter the original data, otherwise
        filter the complete data

        :param complete: the complete dataframe
        :param complete_y: the complete y values
        :return: The result is a dataframe with the filtered data.
        """
        if isinstance(self.filter, ENN):
            original = pd.DataFrame(self.low)
            original_y = pd.DataFrame(self.y)
            result, _ = self.filter.filter_original_complete(
                original, original_y, complete, complete_y
            )
        else:
            result, _ = self.filter.filter(complete, complete_y)
        return result

    def fit(self, samples, y):
        """Fit method."""
        try:
            l, u, y = split(samples, y)
        except IndexError:
            raise ValueError("Dimensions do not match.")

        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        self.__init_values(l, u, y)
        if self.filtering:
            self._fit_stdpnf()
        else:
            self._fit_without()

    def predict(self, src):
        """
        Predict based on a trained classifier.

        :param src: The source image
        :return: The classifier is being returned.
        """
        if self.classifier is None:
            raise AssertionError("The model needs to be fitted first.")
        return self.classifier.predict(src)
