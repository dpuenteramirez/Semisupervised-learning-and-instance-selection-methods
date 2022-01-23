#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    self_training_validation.py
# @Author:      Daniel Puente Ramírez
# @Time:        23/1/22 17:09

import csv
import logging
import os
import glob
import sys
import time
from math import floor
from os import walk

import numpy as np
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.utils import Bunch

time_str = time.strftime("%Y%m%d-%H%M%S")
file_name = 'hyp_self_training'
csv_path_before = os.path.join('tests', file_name + '_pre_' + time_str + '.csv')
csv_path_after = os.path.join('tests', file_name + '_post_' + time_str + '.csv')
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(
                        os.path.join('..', 'logs', '_'.join(
                            [file_name, time_str]))),
                        logging.StreamHandler(sys.stdout)]
                    )

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append(os.path.join(parent, "Instance Selection Algorithms"))

from utils.arff2dataset import arff_data
from ENN import *


def working_datasets(folder):
    if os.path.isdir(folder):
        logging.info(f'Looking up for datasets in {folder}')
    else:
        logging.error(f'{folder} does not exist')

    datasets_found = next(walk(folder), (None, None, []))[2]
    datasets_found.sort()
    logging.info(f'Founded {len(datasets_found)} - {datasets_found}')

    header = ['dataset', 'percent labeled', 'iteration', 'f1-score',
              'mean squared error', 'accuracy score']
    for csv_path in [csv_path_before, csv_path_after]:
        with open(csv_path, 'w') as save:
            w = csv.writer(save)
            w.writerow(header)
            save.close()

    datasets = dict.fromkeys(datasets_found)
    for dataset in datasets_found:
        bunch = arff_data(os.path.join(folder, dataset))
        datasets[dataset] = tuple([bunch['data'], bunch['target']])
    logging.debug('Finalized working datasets')
    return datasets


def self_training_hypothesis(datasets):
    logging.info('Starting hypothesis testing')
    for dataset, (X, y) in datasets.items():
        logging.info(f'Starting dataset: {dataset}')
        for itera in range(5):
            for precision in precisions:
                logging.info(f'\tprecision {precision} - iter {itera + 1}')
                unlabeled_indexes = np.random.choice(len(X), floor(len(X) * (
                        1 - precision)), replace=False)
                labeled_indexes = [i for i in [*range(len(X))] if i not in
                                   unlabeled_indexes]
                y_train = np.copy(y)
                y_train[unlabeled_indexes] = -1
                y_true = y[unlabeled_indexes]
                x_test = X[unlabeled_indexes]
                x_train = X[labeled_indexes]

                svc = SVC(probability=True, gamma="auto")
                model = SelfTrainingClassifier(svc)
                model.fit(X, y_train)
                y_pred = model.predict(x_test)
                y_proba = model.predict_proba(x_test)
                f1 = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
                mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
                acc = accuracy_score(y_true=y_true, y_pred=y_pred)
                logging.info(f'\tpre f1 {f1:.2f} - mse {mse:.2f} - acc'
                             f' {acc:.2f}')

                with open(csv_path_before, 'a') as save:
                    w = csv.writer(save)
                    w.writerow([dataset, precision, itera + 1, f1, mse, acc])
                    save.close()
                logging.debug('\t\tWritten to file before filtering')

                for index0, y_p in enumerate(y_proba):
                    for index1, y_p1 in enumerate(y_p):
                        if y_p1 >= 0.75:
                            y_train[index0] = index1
                            x_train = x_train[:index0] + X[index0] + x_train[
                                index0:]
                            break
                y_labeled = y_train[np.where(y_train != -1)]
                assert len(x_train) == len(y_labeled)
                print(len(x_train))
                dataset_filtered = ENN(Bunch(data=x_train, target=y_labeled), 3)
                x_filtered = dataset_filtered['data']
                y_filtered = dataset_filtered['target']

                # TODO: find x_filtered in X, and replace the corresponding
                #  targets which have been left out with -1. After that,
                #  retrain the model and see if it improves or not.
                breakpoint()


if __name__ == "__main__":
    logging.info('--- Starting ---')
    precisions = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
    hyp_datasets = working_datasets(folder=os.path.join('..', 'datasets'))
    # 'hypothesis'))
    self_training_hypothesis(hyp_datasets)
    logging.info('--- Process completed ---')
