#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ICF_V2.py
# @Author:      Daniel Puente Ramírez
# @Time:        16/12/21 16:35
import copy

from sklearn.datasets import load_iris
from ENN_V2 import ENN
from graficas import grafica_2D
import numpy as np
from sys import maxsize


def __delete_multiple_element__(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)


def coverage_reachable(T):
    """

    :param T: Working dataset.
    :return: list of tuples in the same order of the input samples,
    with (coverage, reachability) of each sample. Cov and Reach are lists
    with the indices of the samples.
    """
    size = len(T.data)
    matrix_distances = np.zeros([size, size])
    distances_to_enemies = []
    sol = []

    # Calculate the distance matrix
    """
    Para cada instancia se calcula la matriz de distancias contra todas las 
    demás, sean de la misma clase o no. shape (n_samples, n_samples)
    Si son de diferente clase se calcula un vector de distancias al enemigo 
    más cercano. shape (n_samples, 1)
    """
    for sample in range(size):
        distance_to_closest_enemy = maxsize
        x_sample = T['data'][sample]
        x_target = T['target'][sample]
        for other_sample in range(size):
            y_sample = T['data'][other_sample]
            y_target = T['target'][other_sample]
            distance = np.linalg.norm(x_sample - y_sample)
            matrix_distances[sample][other_sample] = distance
            # Si son enemigas, nos quedamos con la distancia
            if x_target != y_target and distance < distance_to_closest_enemy:
                distance_to_closest_enemy = distance
        distances_to_enemies.append(distance_to_closest_enemy)

    # Calculate the coverage
    """
    Se calcula el coverage de cada instancia. 
    Conociendo la distancia a la instancia enemiga más cercana, para todas 
    las instancias, excepto una misma, se comprueba si es amiga, si lo es y 
    su distancia es menor que la del enemigo más cercano, se añade al coverage.
    Vector sol con una columna ya, el coverage, son los índices de las 
    muestras que se encuentran dentro.
    """
    for sample in range(size):
        x_coverage = []
        x_target = T['target'][sample]
        distance_to_closest_enemy = distances_to_enemies[sample]
        for other_sample in range(size):
            if sample == other_sample:
                continue
            y_target = T['target'][other_sample]
            if x_target == y_target:
                distance_between_samples = matrix_distances[sample][
                    other_sample]
                if distance_between_samples < distance_to_closest_enemy:
                    x_coverage.append(other_sample)
        sol.append([x_coverage])


    # Calculate the reachable
    """
    Para cada instancia se calcula su reachable sobre el vector solución 
    teniendo en cuenta su coverage.
    I.E. para cada instancia comprobamos si aparece en el coverage de otras 
    instancias. contamos y añadimos en cuáles está.
    """
    for sample in range(size):
        reachable = []
        x_target = T['target'][sample]
        for other_sample in range(size):
            y_target = T['target'][other_sample]
            if sample != other_sample and x_target == y_target:
            # if x_target == y_target:
                coverage = sol[other_sample][0]
                if sample in coverage:
                    reachable.append(other_sample)
        sol[sample].append(reachable)

    return sol


def ICF(X):

    # Wilson
    TS = ENN(X=X, k=3)
    #TS = copy.deepcopy(X)


    while True:
        data = list(TS['data'])
        target = list(TS['target'])
        # Calculate coverage and reachable
        cov_reach = coverage_reachable(TS)

        progress = False
        removable_indexes = []
        for index in range(len(cov_reach)):
            (x_cov, x_reach) = cov_reach[index]
            if len(x_reach) > len(x_cov):
                removable_indexes.append(index)
                progress = True

        __delete_multiple_element__(data, removable_indexes)
        __delete_multiple_element__(target, removable_indexes)
        TS['data'] = data
        TS['target'] = target
        if not progress:
            break

    TS['data'] = np.array(TS['data'])
    return TS


if __name__ == '__main__':
    iris = load_iris()
    print(f'Input samples: {len(iris.data)}')
    S = ICF(iris)
    print(f'Output samples: {len(S.data)}')
    grafica_2D(S)

