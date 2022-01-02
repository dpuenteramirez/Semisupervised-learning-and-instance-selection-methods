#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    DROP3.py
# @Author:      Daniel Puente Ramírez
# @Time:        31/12/21 16:00

from sys import maxsize

import numpy as np
from sklearn.datasets import load_iris

from ENN import ENN
from graficas import grafica_2D


def with_without(x_sample, sample_associates, samples_labels, neighs):
    with_ = 0
    without = 0
    for a_sample in sample_associates:
        a_label = samples_labels[tuple(a_sample)]
        for y_sample, y_neighs, y_labels in neighs:
            if np.array_equal(a_sample, y_sample):
                count = np.bincount(y_labels)
                max_class = np.where(count == np.amax(count))[0][0]
                if max_class == a_label:
                    was_in = False
                    for a_neigh in y_neighs:
                        if np.array_equal(x_sample, a_neigh):
                            was_in = True
                            break
                    if was_in:
                        with_ += 1
                    else:
                        without += 1
                break

    return with_, without


def DROP3(X, k):
    """

    :param X:
    :param k:
    :return:
    """
    # Filtro de ruido
    S = ENN(X, k)

    # Ordenar las instancias en S por la distancia a su enemigo más próximo
    # de más lejano a más cercano
    initial_samples, initial_labels = S['data'], S['target']
    initial_distances = []

    for x_sample, x_label in zip(initial_samples, initial_labels):
        min_distance = maxsize
        for y_sample, y_label in zip(initial_samples, initial_labels):
            if x_label != y_label:
                xy_distance = np.linalg.norm(x_sample - y_sample)
                if xy_distance < min_distance:
                    min_distance = xy_distance
        initial_distances.append([x_sample, x_label, min_distance])

    initial_distances.sort(key=lambda x: x[2], reverse=True)

    # Para cada x en S, encontrar sus k-NN y añadir x a la lista de asociados
    # de sus k-NN
    sample_neighs = []
    sample_associates = [[x, []] for x, _, _ in initial_distances]

    for x_sample, _, _ in initial_distances:
        y_sample_distance = []
        for y_sample, y_label in zip(initial_samples, initial_labels):
            if not np.array_equal(x_sample, y_sample):
                y_sample_distance.append([y_sample, y_label, np.linalg.norm(
                    x_sample - y_sample)])
        y_sample_distance.sort(key=lambda x: x[2])
        x_neighs = [x for x, _, _ in y_sample_distance[:k]]
        x_neighs_labels = [x for _, x, _ in y_sample_distance[:k]]
        sample_neighs.append([x_sample, x_neighs, x_neighs_labels])

        for index, a in enumerate(sample_associates):
            a_sample = a[0]
            for y_sample, _, _ in y_sample_distance[:k]:
                if np.array_equal(a_sample, y_sample):
                    sample_associates[index][1].append(x_sample)
                    break

    # Para cada x en S calcular with and without
    final_samples = [x for x, _, _ in initial_distances]
    final_labels = [x for _, x, _ in initial_distances]
    samples_labels_dict = {tuple(x): y for x, y, _ in initial_distances}
    removed = 0

    for index in range(len(initial_distances)):
        x_sample, x_label = initial_distances[index][0], initial_distances[
            index][1]
        x_associates = sample_associates[index]
        with_, without = with_without(x_sample, x_associates[1],
                                      samples_labels_dict, sample_neighs)

        if without >= with_:
            final_samples = np.delete(final_samples, index - removed, axis=0)
            final_labels = np.delete(final_labels, index - removed, axis=0)

            removed += 1
            for associate in x_associates[1]:
                for index_y, y in enumerate(sample_neighs):
                    y_sample, y_neighs, y_neighs_labels = y
                    if np.array_equal(associate, y_sample):
                        # Eliminar x de la lista de vecinos de a
                        for x_index, neigh in enumerate(y_neighs):
                            if np.array_equal(x_sample, neigh):
                                break
                        del y_neighs[x_index]
                        del y_neighs_labels[x_index]

                        # Encontrar un nuevo vecino para a
                        z_distances = []
                        for z_sample, z_label in zip(final_samples,
                                                     final_labels):
                            if not np.array_equal(associate, z_sample):
                                z_distance = np.linalg.norm(associate -
                                                            z_sample)
                                z_distances.append([z_sample, z_label,
                                                    z_distance])
                        z_distances.sort(key=lambda x: x[2])

                        for neigh_sample, neigh_label, _ in z_distances[:k]:
                            was_in = False
                            for index_z, old_neigh in enumerate(y_neighs):
                                if np.array_equal(neigh_sample, old_neigh):
                                    was_in = True
                                    break
                            if not was_in:
                                y_neighs.append(neigh_sample)
                                y_neighs_labels.append(neigh_label)
                                break
                        sample_neighs[index_y][1] = y_neighs
                        sample_neighs[index_y][2] = y_neighs_labels

                        # Añadir a en la lista de asociados del nuevo vecino
                        for index_z, z in enumerate(sample_associates):
                            z_sample, z_associates = z
                            if np.array_equal(z_sample, neigh_sample):
                                z_associates.append(associate)
                                break
                        sample_associates[index_z][1] = z_associates
                        break

    S['data'] = final_samples
    S['target'] = final_labels.tolist()

    return S


if __name__ == '__main__':
    iris = load_iris()
    print(f'Input samples: {len(iris.data)}')
    S = DROP3(iris, 3)
    print(f'Output samples: {len(S.data)}')
    grafica_2D(S)
