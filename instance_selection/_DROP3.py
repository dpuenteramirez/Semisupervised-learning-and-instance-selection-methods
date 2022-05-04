#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    DROP3.py
# @Author:      Daniel Puente Ramírez
# @Time:        31/12/21 16:00
# @Version:     5.0

import copy
from sys import maxsize

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .utils import transform


class DROP3:
    """
    Wilson, D. R., & Martinez, T. R. (2000). Reduction techniques for
    instance-based learning algorithms. Machine learning, 38(3), 257-286.

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
        Implementation of DROP3.

        The Decremental Reduction Optimization Procedure (DROP) algorithms base
        their selection rule in terms of the partner and associate concept.
        At the very beginning a Wilson Editing algorithm is performed in order
        to remove any noise that may ve contained in the data. Followed by
        the DROP algorithm, in which an instance will be removed is its
        associates are correctly classified without the instance.

        :param samples: DataFrame.
        :param y: DataFrame.
        :return: the input dataset with the remaining samples.
        """
        initial_distances, initial_samples, initial_targets, knn, \
            samples_info = self._create_variables(samples, y)

        self._find_associates(initial_distances, initial_samples,
                              initial_targets, knn, samples_info)

        initial_distances.sort(key=lambda x: x[2], reverse=True)

        removed = 0
        size = len(initial_distances)
        for index_x in range(size):
            x_sample = initial_distances[index_x - removed][0]

            with_, without = self._with_without(tuple(x_sample), samples_info)

            if without >= with_:
                initial_distances = initial_distances[:index_x - removed] + \
                                    initial_distances[index_x - removed + 1:]
                removed += 1

                for a_associate_of_x in samples_info[(tuple(x_sample))][1]:
                    a_neighs, remaining_samples = self._remove_from_neighs(
                        a_associate_of_x, initial_distances,
                        samples_info, x_sample)

                    knn = NearestNeighbors(
                        n_neighbors=self.nearest_neighbors + 2,
                        n_jobs=1, p=self.power_parameter)
                    knn.fit(remaining_samples)
                    _, neigh_ind = knn.kneighbors([a_associate_of_x])
                    possible_neighs = [initial_distances[x][0] for x in
                                       neigh_ind[0]]

                    self._find_new_neighs(a_associate_of_x, a_neighs,
                                          possible_neighs, samples_info)

                    new_neigh = a_neighs[-1]
                    samples_info[tuple(new_neigh)][1].append(
                        a_associate_of_x)

        samples = pd.DataFrame([x for x, _, _ in initial_distances],
                               columns=self.x_attr)
        y = pd.DataFrame([x for _, x, _ in initial_distances])

        return samples, y

    def _create_variables(self, samples, y):
        """
        > It takes in the samples and targets, and returns the initial
        distances, samples, targets, knn, and samples_info

        :param samples: the data
        :param y: the target variable
        :return: initial_distances, initial_samples, initial_targets, knn,
        samples_info
        """
        self.x_attr = samples.keys()
        samples = transform(samples, y)
        s = copy.deepcopy(samples)
        initial_samples = s['data']
        initial_targets = s['target']
        initial_samples, samples_index = np.unique(ar=initial_samples,
                                                   return_index=True, axis=0)
        initial_targets = initial_targets[samples_index]
        knn = NearestNeighbors(n_neighbors=self.nearest_neighbors + 2, n_jobs=1,
                               p=self.power_parameter)
        knn.fit(initial_samples)
        samples_info = {tuple(x): [[], [], y] for x, y in zip(initial_samples,
                                                              initial_targets)}
        initial_distances = []
        return initial_distances, initial_samples, initial_targets, knn, \
            samples_info

    @staticmethod
    def _find_new_neighs(a_associate_of_x, a_neighs, possible_neighs,
                         samples_info):
        """
        > The function takes a sample, finds its neighbors, and then checks if
        any of the neighbors are not already in the list of neighbors. If
        they are not, then they are added to the list of neighbors

        :param a_associate_of_x: the sample we are looking for neighbors for
        :param a_neighs: the list of neighbors of a_associate_of_x
        :param possible_neighs: a list of all the possible neighbors of a given
        point
        :param samples_info: a dictionary with the following structure:
        """
        for pos_neigh in possible_neighs[1:]:
            was_in = False
            for old_neigh in a_neighs:
                if np.array_equal(old_neigh, pos_neigh):
                    was_in = True
                    break
            if not was_in:
                a_neighs.append(pos_neigh)
                break
        samples_info[tuple(a_associate_of_x)][0] = a_neighs

    @staticmethod
    def _remove_from_neighs(a_associate_of_x, initial_distances,
                            samples_info, x_sample):
        """
        > It removes the sample `x_sample` from the list of neighbors of
        `a_associate_of_x` and returns the updated list of neighbors of
        `a_associate_of_x` and the updated list of remaining samples

        :param a_associate_of_x: the sample that is the associate of x
        :param initial_distances: a list of tuples of the form (sample,
        distance, associate)
        :param samples_info: a dictionary of the form
        {(x,y):[neighs,distances,associate]}
        :param x_sample: the sample we want to find the nearest neighbor for
        :return: the new list of neighbors of a_associate_of_x, and the list of
        remaining samples.
        """
        a_neighs = samples_info[tuple(a_associate_of_x)][0]
        index_to_use = 0
        for index_a, neigh in enumerate(a_neighs):
            index_to_use = index_a
            if np.array_equal(neigh, x_sample):
                break

        a_neighs = a_neighs[:index_to_use] + a_neighs[index_to_use + 1:]
        remaining_samples = [x for x, _, _ in initial_distances]

        return a_neighs, remaining_samples

    @staticmethod
    def _find_associates(initial_distances, initial_samples, initial_targets,
                         knn, samples_info):
        """
        For each sample in the initial set, find the closest sample from the
        other class and store it in the initial_distances list

        :param initial_distances: a list of lists, each list containing a
        sample, its target, and its distance to the nearest sample of a
        different class
        :param initial_samples: the samples that we want to find the nearest
        neighbors for
        :param initial_targets: the labels of the initial samples
        :param knn: the k-nearest neighbors model
        :param samples_info: a dictionary that stores the neighbors of each
        sample and the samples that are neighbors of each sample
        """
        for x_sample, x_target in zip(initial_samples, initial_targets):
            min_distance = maxsize
            for y_sample, y_label in zip(initial_samples, initial_targets):
                if x_target != y_label:
                    xy_distance = np.linalg.norm(x_sample - y_sample)
                    if xy_distance < min_distance:
                        min_distance = xy_distance
            initial_distances.append([x_sample, x_target, min_distance])

            _, neigh_ind = knn.kneighbors([x_sample])
            x_neighs = [initial_samples[x] for x in neigh_ind[0][1:]]
            samples_info[tuple(x_sample)][0] = x_neighs

            for neigh in x_neighs[:-1]:
                samples_info[tuple(neigh)][1].append(x_sample)

    @staticmethod
    def _with_without(x_sample, samples_info):
        """
        For each sample in the dataset, we find its associates and then for each
        associate, we find its neighbors. We then find the class with the most
        number of neighbors and compare it with the class of the associate. If
        they are the same, we increment the `with_` variable. If they are not
        the same, we increment the `without` variable

        :param x_sample: the sample we're looking at
        :param samples_info: a dictionary of the form {(x,y):[neighbors,
        associates, target]}
        :return: The number of times the target class of the sample is the most
        common class among its neighbors, with and without the sample itself.
        """
        index_a = 0
        with_ = 0
        without = 0
        x_associates = samples_info[x_sample][1]
        associates_targets = [samples_info[tuple(x)][2] for x in x_associates]
        associates_neighs = [samples_info[tuple(x)][0] for x in x_associates]

        for _, a_target, a_neighs in zip(x_associates,
                                         associates_targets,
                                         associates_neighs):

            neighs_targets = np.ravel(np.array([samples_info[tuple(x)][2] for x
                                                in a_neighs])).astype(int)
            neighs_targets = neighs_targets.tolist()

            count = np.bincount(neighs_targets[:-1])
            max_class = np.where(count == np.amax(count))[0][0]
            if max_class == a_target:
                with_ += 1

            for index_a, neigh in enumerate(a_neighs):
                if np.array_equal(neigh, x_sample):
                    break
            count = np.bincount(neighs_targets[:index_a] + neighs_targets[
                                                           index_a + 1:])
            max_class = np.where(count == np.amax(count))[0][0]
            if max_class == a_target:
                without += 1

        return with_, without
