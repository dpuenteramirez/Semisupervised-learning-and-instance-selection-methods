#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    testing.py
# @Author:      Daniel Puente Ramírez
# @Time:        29/11/21 07:13
import time
import copy
import csv
import sys
import os.path
from os import walk
from statistics import mean

import arff
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import plot_tree
from sklearn.utils import Bunch

from CNN import CNN
from DROP3 import DROP3
from ENN import ENN
from ICF import ICF
from MSS import MSS
from RNN import RNN

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from utils.arff2dataset import arff_data


def main():
    time_str = time.strftime("%Y%m%d-%H%M%S")
    datasets = next(walk('../datasets'), (None, None, []))[2]
    datasets.sort()
    header = ['dataset', 'ENN', 'CNN', 'RNN', 'ICF', 'MSS', 'DROP3']
    csv_path = './test_implementation/test_implementation_' + time_str + '.csv'
    with open(csv_path, 'w') as save:
        w = csv.writer(save)
        w.writerow(header)
        save.close()
    acc = []
    random_state = 0x1122021
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    for path in datasets[5:10]:
        name = path.split('.')[0]
        print(f'Starting {name} dataset...')
        d1 = arff_data(os.path.join('../datasets/', path))
        print(f'\t{len(d1["data"])} samples.')
        current_dataset = [name]
        results_dataset = __evaluate__(dataset=d1, kf=kf)
        acc.append(current_dataset + results_dataset)

    with open(csv_path, 'a') as save:
        w = csv.writer(save)
        w.writerows(acc)
        save.close()


def __evaluate__(dataset, kf):
    algorithms = [DROP3]#[ENN, CNN, RNN, ICF, MSS, DROP3]
    current_dataset = []
    for algorithm in algorithms:
        avg = []
        print(f"\n{algorithm.__name__}________")
        data = copy.copy(dataset)
        X = data['data']
        y = data['target']
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if algorithm.__name__ not in ['ENN', 'DROP3']:
                data_alg = algorithm(X=Bunch(data=X_train, target=y_train))
            else:
                data_alg = algorithm(X=Bunch(data=X_train, target=y_train), k=3)
            avg.append(__train_and_predict__(data_alg, Bunch(data=X_test,
                                                             target=y_test)))
            print(f"\t\titer: {len(avg):>2}. acc: {avg[-1]:.3f}")
        print(f"\taverage: {mean(avg):.2f}")
        current_dataset.append(mean(avg))
    return current_dataset


def __train_and_predict__(data_alg, data):
    # mod_td = DecisionTreeClassifier(max_depth=10, random_state=1)
    mod_td = KNeighborsClassifier(n_neighbors=12, algorithm='brute', p=2,
                                  n_jobs=-1)
    mod_td.fit(data_alg['data'], data_alg['target'])
    prediction = mod_td.predict(data['data'])
    accuracy = metrics.accuracy_score(data['target'], prediction)
    return accuracy


def tree_plot(dataset, tree):
    plt.figure(figsize=(10, 8))
    plot_tree(tree, feature_names=dataset['feature_names'], class_names=dataset[
        'target_names'], filled=True)
    plt.show()


def arff2sk_dataset(dataset_path):
    dataset = arff.load(open(dataset_path, 'r'))
    dat = np.array(dataset['data'])
    tt = np.array(dat[:, -1])
    dat = np.delete(dat, -1, 1)
    dat[dat == ''] = 0.0
    dat = dat.astype(float)

    try:
        tar_names = np.array(dataset['attributes'][-1][1]).astype(int)
        tar = tt.astype(int)
    except ValueError:
        tar_names = np.array([x for x in range(len(dataset['attributes'][-1][
                                                       1]))])
        relation = {}
        for index, target in enumerate(dataset['attributes'][-1][1]):
            relation[target] = index
        tar = np.array([relation[t] for t in tt])

    att_names = np.array([x[0] for x in dataset['attributes'][:-1]])
    dataset = Bunch(data=dat, target=tar, feature_names=att_names,
                    class_names=tar_names)

    return dataset


if __name__ == '__main__':
    main()
