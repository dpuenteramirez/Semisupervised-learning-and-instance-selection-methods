#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    general.py
# @Author:      Daniel Puente Ramírez
# @Time:        24/3/22 10:11

import csv
import logging
import os
import sys
import time
from math import floor
from os import walk

import numpy as np
import pandas as pd
import yagmail
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from instance_selection import ENN, LSSm
from semisupervised import DensityPeaks

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

time_str = time.strftime("%Y%m%d-%H%M%S")
k = 3
folds = 10
precision = 0.05
file_name = "experiments"
csv_results = os.path.join(".", "results", file_name + "_" + time_str + ".csv")
log_file = os.path.join(".", "logs", "_".join([file_name, time_str]) + ".log")

logging.basicConfig(
    level=logging.DEBUG,
    format=" %(asctime)s :: %(levelname)-8s :: %(message)s",
    handlers=[logging.FileHandler(
        log_file), logging.StreamHandler(sys.stdout)],
)


def search_datasets(folder):
    if os.path.isdir(folder):
        logging.info(f"Looking up for datasets in {folder}")
    else:
        logging.error(f"{folder} does not exist")

    datasets_found = next(walk(folder), (None, None, []))[2]
    datasets_found.sort()
    logging.info(f"Founded {len(datasets_found)} - {datasets_found}")

    header = [
        "dataset",
        "percent labeled",
        "fold",
        "base",
        "filter",
        "f1-score",
        "mean squared error",
        "accuracy score",
    ]

    with open(csv_results, "w") as save:
        w = csv.writer(save)
        w.writerow(header)
        save.close()

    datasets = dict.fromkeys(datasets_found)
    for dataset in datasets_found:
        datasets[dataset] = pd.read_csv(
            os.path.join(folder, dataset), header=None)
    logging.debug("Datasets ready to be used")

    return datasets


def main(datasets):
    logging.info("Starting main...")
    random_state = 0x24032022
    skf = StratifiedKFold(n_splits=folds, shuffle=True,
                          random_state=random_state)
    classifiers = [KNeighborsClassifier, DecisionTreeClassifier, GaussianNB]
    classifiers_params = [
        {"n_neighbors": k, "n_jobs": -1},
        {"random_state": random_state},
        {},
    ]
    filters = [ENN, LSSm, "ENANE"]

    for dataset, values in datasets.items():
        logging.info(
            f"\n\nCurrent dataset: {dataset} - Shape: " f"{values.shape}")
        for n_classifier, classifier in enumerate(classifiers):
            classifier_name = classifier.__name__
            for filter_method in filters:
                filter_name = (
                    filter_method
                    if isinstance(filter_method, str)
                    else filter_method.__name__
                )
                samples = values.iloc[:, :-1]
                y = values.iloc[:, -1]
                y_df = pd.DataFrame(y.tolist())
                for fold, (train_index, test_index) in enumerate(skf.split(samples, y)):
                    t_start = time.time()
                    logging.info(
                        f"Dataset: {dataset} -- Classifier: "
                        f"{classifier_name} -- Filter: {filter_name} "
                        f"-- Fold: {fold}"
                    )
                    x_train = samples.iloc[train_index, :].copy(deep=True)
                    x_test = samples.iloc[test_index, :].copy(deep=True)
                    y_train = y_df.iloc[train_index, :].copy(deep=True)
                    y_test = y_df.iloc[test_index, :].copy(deep=True)

                    unlabeled_indexes = np.random.choice(
                        train_index,
                        floor(len(x_train) * (1 - precision)),
                        replace=False,
                    )

                    y_train.at[unlabeled_indexes] = -1

                    model = DensityPeaks.STDPNF(
                        classifier=classifier,
                        classifier_params=classifiers_params[n_classifier],
                        filtering=True,
                        filter_method=filter_method,
                    )
                    try:
                        model.fit(x_train, y_train)
                        y_pred = model.predict(x_test)
                        f1 = f1_score(y_true=y_test, y_pred=y_pred,
                                      average="weighted")
                        mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
                        acc = accuracy_score(y_true=y_test, y_pred=y_pred)

                        logging.info(
                            f"\tf1: {f1:.2f} -- mse: {mse:.2f} -- acc:" f" {acc:.2f}"
                        )
                    except Exception:
                        f1 = mse = acc = ""
                        logging.exception("Failed")
                    t_end = time.time()
                    logging.info(
                        f"\t\tElapsed: {(t_end - t_start) / 60:.2f} minutes")
                    with open(csv_results, "a") as save:
                        w = csv.writer(save)
                        w.writerow(
                            [
                                dataset,
                                precision,
                                fold,
                                classifier_name,
                                filter_name,
                                f1,
                                mse,
                                acc,
                            ]
                        )


if __name__ == "__main__":
    mail = "ntoolsecure"
    passwd = "qfj3nfr_jnt7ATZ8jgh"
    yag = yagmail.SMTP(user=mail, password=passwd)
    t_start_g = time.time()
    try:
        logging.info("--- Starting ---")
        datasets_folder = os.path.join("..", "datasets", "UCI-Experimentation")
        datasets_dfs = search_datasets(datasets_folder)

        main(datasets_dfs)

        logging.info("--- Process completed ---")
        attach = [csv_results, log_file]
        t_end_g = time.time()
        logging.info(f"Elapsed: {(t_end_g - t_start_g) / 60:.2f} minutes")
        yag.send(
            to="dpr1005@alu.ubu.es",
            subject="self_training_validation " "COMPLETED",
            contents="self_training_validation has been completed.\n"
            f"Elapsed: {(t_end_g - t_start_g) / 60:.2f} minutes",
            attachments=attach,
        )
    except Exception as e:
        t_end_g = time.time()
        content = (
            f"FATAL ERROR - Check the attached log\n"
            f"Elapsed: {(t_end_g - t_start_g) / 60:.2f} minutes"
        )

        yag.send(
            to="dpr1005@alu.ubu.es",
            subject="self_training_validation " "ERROR",
            contents=content,
            attachments=[log_file],
        )
        logging.exception("--- Process has broken ---")
        logging.info(f"Elapsed: {(t_end_g - t_start_g) / 60:.2f} minutes")
    logging.info("Email sent successfully")
