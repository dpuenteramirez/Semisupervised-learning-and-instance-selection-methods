#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    DROP3.py
# @Author:      Daniel Puente Ramírez
# @Time:        31/12/21 16:00
import copy
import sys

from sklearn.datasets import load_iris
from graficas import grafica_2D
from sklearn.utils import Bunch
from ENN import ENN
import numpy as np


def with_out(x_sample, x_associates, dataset, k):
    if len(x_associates) == 0:
        return 0, 0, []
    else:
        with_ = 0
        without = 0
        for a_sample, a_label in x_associates:
            samples, labels = dataset['data'], dataset['target']
            distances = []
            for y_sample, y_label in zip(samples, labels):
                if not np.array_equal(a_sample, y_sample):
                    distance = np.linalg.norm(a_sample - y_sample)
                    distances.append([y_sample, y_label, distance])
            distances.sort(key=lambda x: x[2])
            neighbors = distances[:k]
            neighbors_classes = [x for _, x, _ in neighbors]
            neighbors = [x for x, _, _ in neighbors]
            count = np.bincount(neighbors_classes)
            max_class = np.where(count == np.amax(count))[0][0]

            if max_class == a_label:
                x_neigh = False
                for neigh in neighbors:
                    if np.array_equal(x_sample, neigh):
                        x_neigh = True
                if x_neigh:
                    with_ += 1
                else:
                    without += 1
    return with_, without, neighbors


def DROP3(X, k):
    T = ENN(X, k)
    data_samples = T['data']
    data_labels = T['target']

    sorted_samples = []
    for x_sample, x_label in zip(data_samples, data_labels):
        distance_to_closest_enemy = sys.maxsize
        for y_sample, y_label in zip(data_samples, data_labels):
            if x_label != y_label:
                distance = np.linalg.norm(x_sample - y_sample)
                if distance < distance_to_closest_enemy:
                    distance_to_closest_enemy = distance
        sorted_samples.append([x_sample, x_label, distance_to_closest_enemy])
    sorted_samples.sort(key=lambda x: x[2])
    sorted_samples = np.array(sorted_samples)

    associates = [[] for _ in range(len(data_samples))]
    neighbors = [[] for _ in range(len(data_samples))]
    for x_sample, x_label, _ in sorted_samples:
        neighs_index = []
        for index, y in enumerate(sorted_samples):
            if not np.array_equal(x_sample, y[0]):
                distance = np.linalg.norm(x_sample - y[0])
                neighs_index.append([index, distance, y[0], y[1]])
        neighs_index.sort(key=lambda x: x[1])
        neighs = neighs_index[:k]
        for index, _, y_sample, y_label in neighs:
            associates[index].append([x_sample, x_label])
            neighbors[index].append([y_sample, y_label])

    index = 0
    removed = 0
    while index + removed < len(data_samples):
        x_sample, x_label,  _ = sorted_samples[index]
        with_, without, neighbors = with_out(x_sample, associates[index], Bunch(
            data=sorted_samples[:, 0], target=sorted_samples[:, 1]), k)
        if without >= with_:
            sorted_samples = np.delete(sorted_samples, index, axis=0)
            associates_of_neighs = copy.deepcopy(associates[index+removed])
            for i, associates_of_neigh in enumerate(associates_of_neighs):
                a_sample = associates_of_neigh[0]
                a_label = associates_of_neigh[1]
                new_distances = []
                for index2, y in enumerate(sorted_samples):
                    y_sample, _, _ = y
                    if not np.array_equal(a_sample, y_sample):
                        new_distances.append([y_sample, np.linalg.norm(
                            a_sample - y_sample), index2])
                new_distances.sort(key=lambda x: x[1])
                new_neighbors = [x for x, _, _ in new_distances[:k]]
                new_indexes = [x for _, _, x in new_distances[:k]]

                new_neighbor_index = None
                for index2, new_neighbor in enumerate(new_neighbors):
                    was_in = False
                    for old_neighbor in neighbors:
                        if np.array_equal(new_neighbor, old_neighbor):
                            was_in = True
                    if not was_in:
                        new_neighbor_index = new_indexes[index2]

                if new_neighbor_index is not None:
                    associates[new_neighbor_index+removed].append([a_sample,
                                                                   a_label])

            removed += 1
        else:
            index += 1

    T['data'] = np.array([x for x, _, _ in sorted_samples])
    T['target'] = [x for _, x, _ in sorted_samples]

    return T


if __name__ == '__main__':
    iris = load_iris()
    print(f'Input samples: {len(iris.data)}')
    S = DROP3(iris, 3)
    print(f'Output samples: {len(S.data)}')
    grafica_2D(S)
