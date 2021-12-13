#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    test_percent.py
# @Author:      Daniel Puente Ramírez
# @Time:        9/12/21 11:27
import copy
import csv
import os.path
from math import floor
from os import walk
from statistics import mean

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from testing import arff2sk_dataset


def main():
    precisions = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    classifiers = ['KNN', 'Tree']
    datasets = next(walk('../datasets'), (None, None, []))[2]
    datasets.sort()
    header = ['dataset'] + precisions
    random_state = 0x9122021

    for c in classifiers:
        print(f'\n\nClassifier: {c}\n-----------------\n')
        csv_path_acc = f'test_unlabeled/test_unlabeled_cross_validation' \
                       + f'_{c}' + '_acc.csv'
        csv_path_mse = f'test_unlabeled/test_unlabeled_cross_validation' \
                       + f'_{c}' + '_mse.csv'
        with open(csv_path_acc, 'w') as save:
            w = csv.writer(save)
            w.writerow(header)
            save.close()
        with open(csv_path_mse, 'w') as save:
            w = csv.writer(save)
            w.writerow(header)
            save.close()

        for path in datasets[:10]:
            name = path.split('.')[0]
            print(f'Starting {name} dataset...')
            d1 = arff2sk_dataset(os.path.join('../datasets/', path))
            current_dataset_acc = [name]
            current_dataset_mse = [name]

            for precision in precisions:
                print(f"\n\nCurrent precision: {precision * 100:>2}%")
                data = copy.copy(d1)
                print(f'\t{floor(len(data["data"]) * precision)} samples.\t',
                      end='')
                X = data['data']
                y = data['target']

                acc_temp = []
                mse_temp = []
                kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    if precision != 1.0:
                        X_train, _, y_train, _ = \
                            train_test_split(X_train, y_train,
                                             test_size=1 - precision,
                                             random_state=random_state)
                    if c == 'Tree':
                        classifier = DecisionTreeClassifier(max_depth=10,
                                                            random_state=
                                                            random_state)
                    else:
                        classifier = KNeighborsClassifier(n_neighbors=3)
                    classifier.fit(X_train, y_train)
                    prediction = classifier.predict(X_test)
                    acc_temp.append(accuracy_score(prediction, y_test))
                    mse_temp.append(mean_squared_error(prediction, y_test))

                print(f'acc: {mean(acc_temp):.3f}\tmse: {mean(mse_temp):.3f}')
                current_dataset_acc.append(mean(acc_temp))
                current_dataset_mse.append(mean(mse_temp))

            with open(csv_path_acc, 'a') as save:
                w = csv.writer(save)
                w.writerow(current_dataset_acc)
                save.close()
            with open(csv_path_mse, 'a') as save:
                w = csv.writer(save)
                w.writerow(current_dataset_mse)
                save.close()


if __name__ == '__main__':
    main()
