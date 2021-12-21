#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    testing.py
# @Author:      Daniel Puente Ramírez
# @Time:        29/11/21 07:13
import copy
import csv
import os.path
from os import walk
from statistics import mean

import arff
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import Bunch


from CNN import CNN
from ENN import ENN
from ICF_V2 import ICF
from MSS import MSS
from RNN import RNN


def main():
    datasets = next(walk('../datasets'), (None, None, []))[2]
    datasets.sort()
    header = ['dataset', 'ENN', 'CNN', 'RNN', 'ICF', 'MSS']
    csv_path = 'testing_output_cross-validation-KNN19122021.csv'
    with open(csv_path, 'w') as save:
        w = csv.writer(save)
        w.writerow(header)
        save.close()
    acc = []
    random_state = 0x1122021
    kf = KFold(n_splits=2, shuffle=True, random_state=random_state)

    for path in datasets[0:10]:
        name = path.split('.')[0]
        print(f'Starting {name} dataset...')
        d1 = arff2sk_dataset(os.path.join('../datasets/', path))
        print(f'\t{len(d1["data"])} samples.')
        current_dataset = [name]
        results_dataset = __evaluate__(dataset=d1, kf=kf)
        acc.append(current_dataset + results_dataset)

    with open(csv_path, 'a') as save:
        w = csv.writer(save)
        w.writerows(acc)
        save.close()


def __evaluate__(dataset, kf):
    algorithms = [ICF]#[ENN, CNN, RNN, ICF, MSS]
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
            if algorithm.__name__ != "ENN":
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
    accuracy = metrics.accuracy_score(prediction, data['target'])
    return accuracy


def tree_plot(dataset, tree):
    plt.figure(figsize=(10, 8))
    plot_tree(tree, feature_names=dataset['feature_names'], class_names=dataset[
        'target_names'], filled=True)
    plt.show()


def arff2sk_dataset(dataset_path):
    dataset = arff.load(open(dataset_path, 'r'))
    dat = np.array(dataset['data'])
    tar = np.array(dat[:, -1])
    dat = np.delete(dat, -1, 1)
    dat[dat == ''] = 0.0
    dat = dat.astype(float)
    tar_names = np.unique(tar)
    tar_names_list = list(tar_names)
    att_names = np.array([x[0] for x in dataset['attributes'][:-1]])
    tar = np.array([tar_names_list.index(x) for x in tar])
    dataset = Bunch(data=dat, target=tar, feature_names=att_names,
                    class_names=tar_names)

    return dataset


if __name__ == '__main__':
    main()
