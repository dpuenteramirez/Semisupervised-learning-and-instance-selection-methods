#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    testing_unlabeled.py
# @Author:      Daniel Puente Ramírez
# @Time:        6/12/21 09:44

import copy
import csv
import os.path
from math import floor
from os import walk

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import Bunch

from CNN import CNN
from ENN import ENN
from ICF import ICF
from MSS import MSS
from RNN import RNN
from testing import arff2sk_dataset


def main():
    precisions = [0.01, 0.05, 0.1, 0.2, 0.5]
    datasets = next(walk('../datasets'), (None, None, []))[2]
    datasets.sort()
    header = ['dataset', 'ENN', 'CNN', 'RNN', 'ICF', 'MSS']
    for precision in precisions:
        print(f"\n\nCurrent precision: {precision}")
        csv_path_acc = f'testing_out/test_unlabeled_{precision}_acc.csv'
        csv_path_mse = f'testing_out/test_unlabeled_{precision}_mse.csv'
        with open(csv_path_acc, 'w') as save:
            w = csv.writer(save)
            w.writerow(header)
            save.close()
        with open(csv_path_mse, 'w') as save:
            w = csv.writer(save)
            w.writerow(header)
            save.close()
        acc = []
        mse = []
        random_state = 0x1122021

        for path in datasets[:10]:
            name = path.split('.')[0]
            print(f'Starting {name} dataset...')
            d1 = arff2sk_dataset(os.path.join('../datasets/', path))
            print(f'\t{floor(len(d1["data"]) * precision)} samples.')
            current_dataset = [name]
            results_dataset = np.array(__evaluate__(dataset=d1,
                                                    precision=precision,
                                                    random_state=random_state))

            acc.append(current_dataset + list(results_dataset[:, 0]))
            mse.append(current_dataset + list(results_dataset[:, 1]))

        with open(csv_path_acc, 'a') as save:
            w = csv.writer(save)
            w.writerows(acc)
            save.close()
        with open(csv_path_mse, 'a') as save:
            w = csv.writer(save)
            w.writerows(acc)
            save.close()


def __evaluate__(dataset, precision, random_state):
    algorithms = [ENN, CNN, RNN, ICF, MSS]
    current_dataset = []
    for algorithm in algorithms:
        print(f"\n{algorithm.__name__}________")
        data = copy.copy(dataset)
        X = data['data']
        y = data['target']
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, train_size=precision,
                             random_state=random_state, stratify=y)

        if algorithm.__name__ != "ENN":
            data_alg = algorithm(X=Bunch(data=X_train, target=y_train))
        else:
            data_alg = algorithm(X=Bunch(data=X_train, target=y_train), k=3)

        if len(data_alg['data']) != 0:
            acc, mse = __train_and_predict__(data_alg,
                                             Bunch(data=X_test, target=y_test))
            print(f"\t|-> acc: {acc:.3f}\n\t|-> mse: {mse:.3f}")
        else:
            acc = '-'
            mse = '-'
            print(f"\t|-> acc: {acc}\n\t|-> mse: {mse}")

        current_dataset.append([acc, mse])
    return current_dataset


def __train_and_predict__(data_alg, data_test):
    tree = DecisionTreeClassifier(max_depth=10, random_state=1)
    tree.fit(data_alg['data'], data_alg['target'])
    prediction = tree.predict(data_test['data'])
    acc = accuracy_score(prediction, data_test['target'])
    mse = mean_squared_error(prediction, data_test['target'])
    return acc, mse


if __name__ == '__main__':
    main()
