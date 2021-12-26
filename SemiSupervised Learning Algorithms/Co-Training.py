#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    Co-Training.py
# @Author:      Daniel Puente Ramírez
# @Time:        22/12/21 09:27

import argparse
import sys
from math import floor
from os import walk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from tqdm import trange

sys.path.insert(0, "../utils")
from arff2dataset import arff2sk_dataset


def co_training(p, n, k, u, L, U):
    tot = p + n
    counts = {}
    for tar in L['class_names']:
        counts[tar] = 0
    for tar in L['target']:
        counts[tar] += 1

    p_class = max(counts, key=counts.get)

    rng = np.random.default_rng()
    model = RandomForestClassifier(n_jobs=-1)

    __U__ = rng.choice(U['data'], size=u, replace=False, shuffle=False)

    for _ in trange(k):

        size = floor(len(L['data']) / 2)
        rem = len(L['data']) - size
        x1, y1 = L['data'][:size], L['target'][:size]
        x2, y2 = L['data'][rem:], L['target'][rem:]
        h1 = model.fit(x1, y1)
        h2 = model.fit(x2, y2)

        pred_1, pred_proba_1 = h1.predict(__U__), h1.predict_proba(__U__)
        pred_2, pred_proba_2 = h2.predict(__U__), h2.predict_proba(__U__)

        idx_1 = np.amax(pred_proba_1, axis=1)
        aa = pd.DataFrame(pred_proba_1)
        idx_2 = np.amax(pred_proba_2, axis=1)
        prediction_1 = [[x, y, z, u_] for x, y, z, u_ in zip(pred_1,
                                                             pred_proba_1,
                                                             idx_1, __U__)]
        prediction_2 = [[x, y, z, u_] for x, y, z, u_ in zip(pred_2,
                                                             pred_proba_2,
                                                             idx_2, __U__)]

        prediction_1.sort(key=lambda x: x[2], reverse=True)
        prediction_2.sort(key=lambda x: x[2], reverse=True)

        predictions = np.concatenate((prediction_1, prediction_2), axis=0)

        predictions_temp = pd.DataFrame(predictions, columns=['Class', 'Array',
                                                              'Confidence',
                                                              'Sample'])
        predictions = []

        p_added = 0
        for _, pred in predictions_temp.iterrows():
            if len(predictions) == tot:
                break
            if pred['Class'] == p_class and p_added < p:
                predictions.append(pred)
                p_added += 1
            elif pred['Class'] != p_class:
                predictions.append(pred)

        predictions = pd.DataFrame(predictions, columns=['Class', 'Array',
                                                         'Confidence',
                                                         'Sample'])

        samples = np.array([x for x in predictions.pop('Sample').to_numpy()],
                           dtype=object)
        classes = np.array([x for x in predictions.pop('Class').to_numpy()])

        found = []
        for index, s1 in enumerate(U['data']):
            for s2 in samples:
                if np.array_equal(s1, s2):
                    found.append(index)
                    break
        U['data'] = np.delete(U['data'], found, axis=0)

        found = []
        for index, s1 in enumerate(__U__):
            for s2 in samples:
                if np.array_equal(s1, s2):
                    found.append(index)
                    break

        __U__ = np.delete(__U__, found, axis=0)

        L['data'] = np.concatenate((L['data'], samples), axis=0)
        L['target'] = np.concatenate((L['target'], classes), axis=0)
        try:
            temp = rng.choice(U['data'], size=int(2 * p + 2 * n), shuffle=False)
        except ValueError:
            # If no more values remain for taking, it was not correctly
            # parametrized
            return L['data'], L['target']

        __U__ = np.vstack([__U__, temp])

    return L['data'], L['target']


if __name__ == '__main__':
    folder = '../datasets/'

    files = next(walk(folder), (None, None, []))[2]
    files_tot = '\t'.join(files)
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', nargs=1)
    args = parser.parse_args()

    # if args.d[0] not in files_tot:
    #     print('File does not exist.')
    #     exit(1)
    # dataset = [file for file in files if args.d[0] in file]
    # if len(dataset) != 1:
    #     print('More than one file found.')
    #     exit(2)
    dataset = ['iris.arff']

    dataset = arff2sk_dataset(folder + dataset[0])

    target_labels = [int(float(x)) for x in dataset.class_names]
    y = dataset.target
    X = dataset.data
    X_test, X_train, y_test, y_train = train_test_split(X, y,
                                                        train_size=0.95,
                                                        stratify=y)
    X = Bunch(data=X_train, target=y_train, class_names=target_labels)
    y = Bunch(data=X_test)

    data, target = co_training(p=3, n=9, k=30, u=20, L=X, U=y)

    model = RandomForestClassifier(n_jobs=-1)
    model.fit(data, target)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, labels=target_labels))

    cm = confusion_matrix(y_test, y_pred, labels=target_labels)
    dsp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                 display_labels=target_labels)
    dsp.plot()
    plt.show()
