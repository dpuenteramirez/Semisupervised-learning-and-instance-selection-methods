#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    Co-Training.py
# @Author:      Daniel Puente Ramírez
# @Time:        22/12/21 09:27

import sys
from math import floor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import Bunch
from tqdm import trange
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

sys.path.insert(0, "../utils")
from arff2dataset import arff2sk_dataset


def co_training(p, n, k, u, L, U):
    tot = p + n

    rng = np.random.default_rng()

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1)

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
        idx_2 = np.amax(pred_proba_2, axis=1)
        prediction_1 = [[x, y, z, u] for x, y, z, u in zip(pred_1, pred_proba_1,
                                                           idx_1, __U__)]
        prediction_2 = [[x, y, z, u] for x, y, z, u in zip(pred_2, pred_proba_2,
                                                           idx_2, __U__)]

        prediction_1.sort(key=lambda x: x[2], reverse=True)
        prediction_2.sort(key=lambda x: x[2], reverse=True)

        predictions = np.concatenate((prediction_1[:tot], prediction_2[
                                                          :tot]), axis=0)

        predictions = pd.DataFrame(predictions, columns=['Class', 'Array',
                                                         'Confidence',
                                                         'Sample'])
        samples = np.array([x for x in predictions.pop('Sample').to_numpy()],
                           dtype=object)
        classes = np.array([x for x in predictions.pop('Class').to_numpy()])

        L['data'] = np.concatenate((L['data'], samples), axis=0)
        L['target'] = np.concatenate((L['target'], classes), axis=0)

        __U__ = rng.choice(U['data'], size=int(2 * p + 2 * n), replace=False,
                           shuffle=False)

    return L['data'], L['target']


if __name__ == '__main__':
    # dataset = load_iris()
    #dataset = arff2sk_dataset('../datasets/08_segment_norm.arff')
    #dataset = arff2sk_dataset('../datasets/03_contraceptive_norm.arff')
    dataset = arff2sk_dataset('../datasets/25_coil2000_norm.arff')
    # dataset = arff2sk_dataset('../datasets/16_page-blocks_norm.arff')

    target_labels = [int(x) for x in dataset.class_names]

    y = dataset.target
    X = dataset.data
    X_test, X_train, y_test, y_train = train_test_split(X, y, train_size=0.75)
    X = Bunch(data=X_train, target=y_train)
    y = Bunch(data=X_test)

    data, target = co_training(p=1, n=3, k=20, u=10, L=X, U=y)
    model = KNeighborsClassifier(n_jobs=-1, n_neighbors=1)
    model.fit(data, target)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, labels=target_labels))

    cm = confusion_matrix(y_test, y_pred, labels=target_labels)
    dsp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                 display_labels=target_labels)
    dsp.plot()
    plt.show()
