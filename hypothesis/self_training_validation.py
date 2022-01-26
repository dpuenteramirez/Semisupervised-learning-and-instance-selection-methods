#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    self_training_validation.py
# @Author:      Daniel Puente Ramírez
# @Time:        23/1/22 17:09

import csv
import logging
import os
import sys
import time
from math import floor
from os import walk

import numpy as np
import yagmail
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.utils import Bunch

threshold = 0.75
k = 3
time_str = time.strftime("%Y%m%d-%H%M%S")
file_name = 'hyp_self_training'
log_file = os.path.join('..', 'logs', '_'.join([file_name, time_str]) + '.log')
csv_path = os.path.join('tests', file_name + '_' + time_str
                        + '.csv')

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(log_file),
                              logging.StreamHandler(sys.stdout)]
                    )

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append(os.path.join(parent, "Instance Selection Algorithms"))

from utils.arff2dataset import arff_data
from ENN import *
from ENN_self_training import *


def working_datasets(folder):
    if os.path.isdir(folder):
        logging.info(f'Looking up for datasets in {folder}')
    else:
        logging.error(f'{folder} does not exist')

    datasets_found = next(walk(folder), (None, None, []))[2]
    datasets_found.sort()
    logging.info(f'Founded {len(datasets_found)} - {datasets_found}')

    header = [
        'dataset',
        'percent labeled',
        'fold',
        'f1-score before',
        'mean squared error before',
        'accuracy score before',
        'f1-score after with deletion',
        'mean squared error after with deletion',
        'accuracy score after with deletion',
        'f1-score after without deletion',
        'mean squared error after without deletion',
        'accuracy score after without deletion',
        'initial samples',
        'samples after self-training',
        'samples after filtering with deletion',
        'samples after filtering without deletion'
    ]

    with open(csv_path, 'w') as save:
        w = csv.writer(save)
        w.writerow(header)
        save.close()

    datasets = dict.fromkeys(datasets_found)
    for dataset in datasets_found:
        bunch = arff_data(os.path.join(folder, dataset))
        datasets[dataset] = tuple([bunch['data'], bunch['target']])
    logging.debug('Datasets ready to be used')
    return datasets


def training_model(x_train, y_train, x_test, y_test, csv_output, pre):
    logging.debug('\t\tCreating model')
    svc = SVC(probability=True, gamma="auto")
    model = SelfTrainingClassifier(svc, threshold=threshold)
    logging.debug('\t\tFitting model')
    try:
        model.fit(x_train, y_train)
        fit_ok = True
    except ValueError:
        fit_ok = False
        logging.exception('Error while fitting')
    if fit_ok:
        logging.debug('\t\tPredicting')
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)
        f1 = f1_score(y_true=y_test, y_pred=y_pred, average="weighted")
        mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
        acc = accuracy_score(y_true=y_test, y_pred=y_pred)
        logging.info(
            f'\t{"pre" if pre else "post"} f1 {f1:.2f} - mse {mse:.2f} - '
            f'acc {acc:.2f}')
    else:
        f1 = mse = acc = ''
        y_proba = None

    csv_output += f1, mse, acc

    return y_proba, fit_ok, csv_output


def self_training_hypothesis(datasets):
    logging.info('Starting hypothesis testing')
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    for dataset, (X, y) in datasets.items():
        logging.info(f'Current dataset: {dataset} - Total samples: {len(X)}')
        if len(X) != len(set([tuple(i) for i in X])):
            logging.warning('\tThe dataset contains repeated samples')

        for precision in precisions:
            for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
                csv_output = [dataset, precision, fold]
                logging.info(f'\tprecision {precision} - iter {fold + 1}')
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                unlabeled_indexes = np.random.choice(len(x_train), floor(len(
                    x_train) * (1 - precision)), replace=False)
                labeled_indexes = [i for i in [*range(len(x_train))] if i
                                   not in unlabeled_indexes]

                samples_before = len(labeled_indexes)
                logging.debug(f'\t\tSamples before: {samples_before}')
                y_modified = np.copy(y_train)
                x_labeled = x_train[labeled_indexes]
                y_labeled = y_modified[labeled_indexes]
                y_modified[unlabeled_indexes] = -1

                y_proba, fit_ok, csv_output = training_model(
                    x_train, y_modified, x_test, y_test, csv_output, True)

                if not fit_ok:
                    logging.warning(f'Fold {fold} failed with this precision.')
                    csv_output += ['', '', '', '', '', '',
                                   samples_before, '', '', '']

                    with open(csv_path, 'a') as save:
                        w = csv.writer(save)
                        w.writerow(csv_output)
                        save.close()
                    continue
                else:
                    logging.debug('\t\tBefore - done')

                samples_after_sl = samples_before
                x_labeled_before = np.copy(x_labeled)
                y_labeled_before = np.copy(y_labeled)
                for index0, y_p in enumerate(y_proba):
                    for index1, y_p1 in enumerate(y_p):
                        if y_p1 >= threshold:
                            samples_after_sl += 1
                            y_labeled = np.concatenate((y_labeled, [index1]))
                            x_labeled = np.concatenate((x_labeled, [x_train[
                                                                        index0]]))
                            break

                logging.debug(f'\t\tSamples after SL: {samples_after_sl}')

                try:
                    assert len(x_labeled) == len(y_labeled)
                except AssertionError:
                    logging.fatal(f'len(x_labeled) != len(y_labeled) -'
                                  f' {len(x_labeled)} != {len(y_labeled)}')
                    exit(1)

                logging.debug('\t\tFiltering with deletion')
                try:
                    dataset_filtered_deleting = ENN(Bunch(data=x_labeled,
                                                          target=y_labeled), k)
                    logging.debug('\t\tFiltered')
                except ValueError:
                    dataset_filtered_deleting = None
                    logging.warning(f'Expected n_neighbors <= n_samples,  '
                                    f'but n_samples = {len(x_labeled)}, '
                                    f'n_neighbors = {k}')

                logging.debug('\t\tFiltering without deletion')
                dataset_filtered_no_deleting = ENN_for_self_training(
                    Bunch(data=x_labeled_before, target=y_labeled_before),
                    Bunch(data=x_labeled, target=y_labeled), k
                ) if len(x_labeled) > len(x_labeled_before) else None

                if dataset_filtered_no_deleting is not None:
                    logging.debug('\t\tFiltered')

                if dataset_filtered_deleting is not None:
                    logging.debug('\t\tStarting with the deletion model')
                    y_after_filtering = np.copy(y_train)
                    samples_after_filtering_with_deletion = len(
                        dataset_filtered_deleting['data'])
                    logging.debug(f'\t\tSamples after filtering with deletion:'
                                  f' {samples_after_filtering_with_deletion}')
                    x_samples_filtered = dataset_filtered_deleting['data']

                    indexes = []
                    for index0, x_sample in enumerate(x_train):
                        label_remains = False
                        for index1, y_sample in enumerate(x_samples_filtered):
                            if np.array_equal(x_sample, y_sample):
                                indexes.append(index0)
                                break
                    indexes_to_remove = [x for x in [*range(len(x_train))]
                                         if x not in indexes]
                    y_after_filtering[indexes_to_remove] = -1

                    logging.debug('\t\tDataset ready to train the new model')
                    _, _, csv_output = training_model(
                        x_train, y_after_filtering, x_test, y_test,
                        csv_output, False)

                else:
                    csv_output += ['', '', '']
                    samples_after_filtering_with_deletion = ''

                if dataset_filtered_no_deleting is not None:
                    logging.debug('\t\tStarting with the non deletion model')
                    y_after_filtering = np.copy(y_train)
                    samples_after_filtering_without_deletion = len(
                        dataset_filtered_no_deleting['data'])
                    logging.debug('\t\tSamples after filtering without '
                                  'deletion '
                                  f'{samples_after_filtering_without_deletion}')
                    x_samples_filtered = dataset_filtered_no_deleting['data']

                    indexes = []
                    for index0, x_sample in enumerate(x_train):
                        label_remains = False
                        for index1, y_sample in enumerate(x_samples_filtered):
                            if np.array_equal(x_sample, y_sample):
                                indexes.append(index0)
                                break
                    indexes_to_remove = [x for x in [*range(len(x_train))]
                                         if x not in indexes]
                    y_after_filtering[indexes_to_remove] = -1

                    logging.debug('\t\tDataset ready to train the new model')
                    _, _, csv_output = training_model(
                        x_train, y_after_filtering, x_test, y_test,
                        csv_output, False)

                else:
                    csv_output += ['', '', '']
                    samples_after_filtering_without_deletion = ''

                csv_output += [samples_before, samples_after_sl,
                               samples_after_filtering_with_deletion,
                               samples_after_filtering_without_deletion]

                with open(csv_path, 'a') as save:
                    w = csv.writer(save)
                    w.writerow(csv_output)
                    save.close()

                logging.debug('\t\tWritten to file.')
                logging.info('\n\n')


if __name__ == "__main__":
    yag = yagmail.SMTP(user=<SENDER EMAIL>, password=<SENDER PASSWORD>)
    try:
        logging.info('--- Starting ---')
        precisions = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]
        hyp_datasets = working_datasets(folder=os.path.join('..', 'datasets',
                                                            'hypothesis'))

        self_training_hypothesis(hyp_datasets)

        logging.info('--- Process completed ---')
        attach = [csv_path]
        yag.send(to=<RECIPIENT EMAIL>, subject='self_training_validation '
                                                  'COMPLETED',
                 contents='self_training_validation has been completed.',
                 attachments=attach)
    except Exception as e:
        content = f'FATAL ERROR - Check the attached log'

        yag.send(to='dpr1005@alu.ubu.es', subject='self_training_validation '
                                                  'ERROR',
                 contents=content, attachments=[log_file])
        logging.exception('--- Process has broken ---')
    logging.info("Email sent successfully")
