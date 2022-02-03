#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    test_implementation_semisupervised.py
# @Author:      Daniel Puente Ramírez
# @Time:        5/1/22 15:54
import csv
import logging
import os
import sys
import time
from os import walk
from os.path import join
from statistics import mean

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from semisupervised.CoTraining import CoTraining
from semisupervised.DemocraticCoLearning import DemocraticCoLearning
from semisupervised.TriTraining import TriTraining
from utils.arff2dataset import arff_data
from utils.threads import ReturnValueThread


def test_implementation():
    time_str = time.strftime("%Y%m%d-%H%M%S")
    datasets = next(walk('../datasets'), (None, None, []))[2]
    datasets.sort()
    header = ['dataset', 'Co-Training', 'Tri-Training', 'Democratic '
                                                        'Co-Learning']
    csv_path = './tests/test_implementation_' + time_str + '.csv'
    with open(csv_path, 'w') as save:
        w = csv.writer(save)
        w.writerow(header)
        save.close()

    random_state = 0x06012022
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for path in datasets[:10]:
        name = path.split('.')[0]
        print(f'Starting {name} dataset...')
        d1 = arff_data(join('../datasets/', path))
        print(f'\t{len(d1["data"])} samples.')
        current_dataset = [name]
        results_dataset = __evaluate__(dataset=d1, kf=kf,
                                       random_state=random_state,
                                       header=header[1:])
        acc = current_dataset + results_dataset

        with open(csv_path, 'a') as save:
            w = csv.writer(save)
            w.writerow(acc)
            save.close()


def __compute__(name, n_alg, X, y, kf, random_state):
    logging.info(f'\tStarting: {name}')
    avg = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if n_alg == 0:
            model = CoTraining(random_state=random_state, u=10, k=5)
        elif n_alg == 1:
            model = TriTraining(random_state=random_state, learn=4)
        else:
            model = DemocraticCoLearning(random_state=random_state)
        model.fit(L=X_train, U=X_test, y=y_train)
        y_pred = model.predict(X_test)
        avg.append(accuracy_score(y_test, y_pred))
        logging.info(f"\t\t{name}: {len(avg):>2}. acc: {avg[-1]:.3f}")

    logging.info(f'\tFinished: {name}')
    return mean(avg)


def __evaluate__(dataset, kf, random_state, header):
    format_ = "%(asctime)s: %(message)s"

    logging.basicConfig(format=format_, level=logging.INFO, datefmt="%H:%M:%S")

    current_dataset = []

    X = dataset['data']
    y = dataset['target']

    logging.info(f"\tMain   : {header[0]}")
    th_co = ReturnValueThread(target=__compute__, args=(header[0], 0, X, y,
                                                        kf, random_state))
    logging.info(f"\tMain   : {header[1]}")
    th_tri = ReturnValueThread(target=__compute__, args=(header[1], 1, X, y,
                                                         kf, random_state))
    logging.info(f"\tMain   : {header[2]}")
    th_demo = ReturnValueThread(target=__compute__, args=(header[2], 2, X, y,
                                                          kf, random_state))
    # t = __compute__(header[2], 2, X, y, kf, random_state)
    th_co.start()
    th_tri.start()
    th_demo.start()
    co_acc = th_co.join()
    tri_acc = th_tri.join()
    demo_acc = th_demo.join()

    current_dataset.append(co_acc)
    current_dataset.append(tri_acc)
    current_dataset.append(demo_acc)

    return current_dataset


if __name__ == '__main__':
    test_implementation()
