#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    self_training.py
# @Author:      Daniel Puente Ramírez
# @Time:        10/12/21 08:36
from os import walk
import os.path
from testing import arff2sk_dataset
from math import floor
import copy
import csv
import numpy as np
from statistics import mean
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.neighbors import KNeighborsClassifier


def main():
    precisions = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    datasets = next(walk('../datasets'), (None, None, []))[2]
    datasets.sort()
    random_state = 0x10122021

    csv_path = f'test_self_training/accuracy_self_training.csv'
    header = ['dataset'] + precisions

    with open(csv_path, 'w') as save:
        w = csv.writer(save)
        w.writerow(header)
        save.close()

    for path in datasets[:10]:
        name = path.split('.')[0]
        print(f'Starting {name} dataset...')
        d1 = arff2sk_dataset(os.path.join('../datasets/', path))

        current_dataset = [name]

        for precision in precisions:
            print(f"\n\nCurrent precision: {precision * 100:>2}%")
            data = copy.copy(d1)
            print(f'\t{floor(len(data["data"]) * precision)} samples.\t',
                  end='\n')

            X = data['data']
            y = data['target']

            acc_temp = []
            kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                if precision != 1.0:
                    indexes = np.random.choice(len(y_train),
                                               floor(len(y_train) * (
                                                           1 - precision)),
                                               replace=False)
                    indexes = np.setdiff1d(np.arange(len(y_train)), indexes)
                    y_train[indexes] = -1
                    X_train, _, y_train, _ = \
                        train_test_split(X_train, y_train,
                                         test_size=1 - precision,
                                         random_state=random_state)

                neigh = KNeighborsClassifier(n_neighbors=3)
                classifier = SelfTrainingClassifier(neigh, verbose=True)
                classifier.fit(X_train, y_train)
                prediction = classifier.predict(X_test)
                acc_temp.append(accuracy_score(prediction, y_test))
                print(f'acc: {mean(acc_temp):.3f}')

            current_dataset.append(mean(acc_temp))
        with open(csv_path, 'a') as save:
            w = csv.writer(save)
            w.writerow(current_dataset)
            save.close()


if __name__ == '__main__':
    main()