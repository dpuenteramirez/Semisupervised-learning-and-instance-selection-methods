#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    CNN.py
# @Author:      Daniel Puente Ramírez
# @Time:        19/11/21 07:13

import random
import numpy as np
from sklearn.datasets import load_iris
from graficas import grafica_2D
from ENN import ENN


def __delete_multiple_element__(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)


def CNN(X, k=3):
    """
    Comienza seleccionando un objeto aleatorio de cada clase.
    Para todos los objetos se los intenta clasificar.
    :param k:
    :param X:
    :return:
    """

    assert k > 0, "k must be greater than 0. Default value is 3."
    store = []
    store_classes = []
    for i in range(k):
        random_int = random.randint(0, len(X['data'] - 1))
        store.append(X['data'][random_int])
        store_classes.append(X['target'][random_int])
    handbag = []

    for sample_class, sample in zip(X.target, X.data):
        # NN Rule classification: assigns an unclassified sample to the same
        # class as the nearest of n stored, correctly samples.
        # Distancia Euclídea
        euc = []
        for s in store:
            euc.append(np.linalg.norm(s - sample))
        euc = np.array(euc)
        euc_nn = np.amin(euc)
        index_nn = np.ravel(np.where(euc == euc_nn))
        nn_class = store_classes[index_nn[0]]

        # With the NN and its class, we step forward into looking if the
        # classification of the classes match or not
        if nn_class == sample_class:
            handbag.append((sample_class, sample))
        else:
            store.append(sample)
            store_classes.append(sample_class)
    """ 
    After one pass through the original sample set, the procedure continues to 
    loop trough *handbag* until termination.
        - Handbag exhausted. All members are in store.
        - One complete pass is made with no transfers to store. 
    """
    store_not_modified = True
    while len(handbag) > 0 and store_not_modified:
        store_not_modified = False
        indexes = []
        for index, s2 in enumerate(handbag):
            sample_class, sample = s2
            euc = []
            for s in store:
                euc.append(np.linalg.norm(s - sample))
            euc = np.array(euc)
            euc_nn = np.amin(euc)
            index_nn = np.ravel(np.where(euc == euc_nn))
            nn_class = store_classes[index_nn[0]]
            if nn_class != sample_class:
                store.append(sample)
                store_classes.append(sample_class)
                indexes.append(index)
                store_not_modified = True
        __delete_multiple_element__(handbag, indexes)
    del handbag

    # Save changes in the original dataset
    store = np.array(store)
    X['data'] = store
    X['target'] = store_classes
    print(f"{len(store)} samples retrieved.")
    return X


def main():
    data = load_iris()
    k = 3
    S = CNN(data)
    grafica_2D(S)


if __name__ == '__main__':
    main()
