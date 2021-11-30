#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    testing.py
# @Author:      Daniel Puente Ramírez
# @Time:        29/11/21 07:13
import copy
import os.path

import arff
import csv
from os import walk
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.utils import Bunch
import matplotlib.pyplot as plt
import numpy as np
from ENN import ENN
from CNN import CNN
from RNN import RNN
from ICF import ICF
from MSS import MSS


def main():
    datasets = next(walk('../datasets'), (None, None, []))[2]
    datasets.sort()
    header = ['dataset', 'ENN', 'CNN', 'RNN', 'ICF', 'MSS']
    acc = []
    for path in datasets[:10]:
        name = path.split('.')[0]
        print(f'Starting {name} dataset...')
        d1 = arff2sk_dataset(os.path.join('../datasets/', path))
        print(f'\t{len(d1["data"])} samples.')
        row = [name]

        print('\tENN...')
        data = copy.copy(d1)
        data_alg = ENN(X=data, k=3)
        print(f"\t\t{len(data_alg['data'])} final samples.")
        row.append(__train_and_predict__(data_alg, data))

        print('\tCNN...')
        data = copy.copy(d1)
        data_alg = CNN(X=data)
        print(f"\t\t{len(data_alg['data'])} final samples.")
        row.append(__train_and_predict__(data_alg, data))

        print('\tRNN...')
        data = copy.copy(d1)
        data_alg = RNN(X=data)
        print(f"\t\t{len(data_alg['data'])} final samples.")
        row.append(__train_and_predict__(data_alg, data))

        print('\tICF...')
        data = copy.copy(d1)
        data_alg = ICF(X=data)
        print(f"\t\t{len(data_alg['data'])} final samples.")
        row.append(__train_and_predict__(data_alg, data))

        print('\tMSS...')
        data = copy.copy(d1)
        data_alg = MSS(X=data)
        print(f"\t\t{len(data_alg['data'])} final samples.")
        row.append(__train_and_predict__(data_alg, data))

        acc.append(row)
        break
    csv_path = './testing_output.csv'
    with open(csv_path, 'w') as save:
        w = csv.writer(save)
        w.writerow(header)
        w.writerows(acc)


def __train_and_predict__(data_alg, data):
    mod_td = DecisionTreeClassifier(max_depth=3, random_state=1)
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
