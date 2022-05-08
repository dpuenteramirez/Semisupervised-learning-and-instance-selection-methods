#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    SemiSupervised.py
# @Author:      Daniel Puente Ramírez
# @Time:        16/4/22 00:22

import random

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris as load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from instance_selection import ENN
from semisupervised import STDPNF, CoTraining, TriTraining, \
    DemocraticCoLearning


@pytest.fixture
def digits_dataset_ss():
    """
    It loads the digits dataset, splits it into train and test sets, and then
    randomly assigns 55% of the training set as unlabeled
    :return: x_train, x_test, y_train, y_test, opt_labels
    """
    x, y = load_digits(return_X_y=True, as_frame=True)
    x = x.to_numpy()
    y = y.to_numpy()
    opt_labels = np.unique(y)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.75, stratify=y, random_state=42
    )
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    li = list(set(range(x_train.shape[0])))
    unlabeled = random.sample(li, int(x_train.shape[0] * 0.55))
    y_train.loc[unlabeled] = -1

    return x_train, x_test, y_train, y_test, opt_labels


def base(x_train, x_test, y_train, y_test, opt_labels, algorithm, params=None):
    """
    It takes in a training and testing set, a list of possible labels, and a
    model, and checks the predictions of the model on the testing set

    :param x_train: The training data
    :param x_test: The test data
    :param y_train: The training labels
    :param y_test: the actual labels of the test set
    :param opt_labels: the set of labels that the model can predict
    :param algorithm: the algorithm to use
    :param params: a dictionary of parameters to pass to the algorithm
    """
    assert isinstance(x_train, pd.DataFrame) and isinstance(y_train,
                                                            pd.DataFrame)
    assert isinstance(x_test, pd.DataFrame) and isinstance(y_test,
                                                           pd.DataFrame)
    model = algorithm(**params) if params is not None else algorithm()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    assert set(y_pred).issubset(opt_labels)


def test_co_training(digits_dataset_ss):
    """
    It tests the Co-Training algorithm on the digits dataset

    :param digits_dataset_ss: The dataset we're using
    """
    x_train, x_test, y_train, y_test, opt_labels = digits_dataset_ss
    base(x_train, x_test, y_train, y_test, opt_labels, CoTraining,
         {'p': 1, 'n': 3, 'k': 1, 'u': 7})
    base(x_train, x_test, y_train, y_test, opt_labels, CoTraining,
         {'p': 1, 'n': 3, 'k': 1, 'u': 7,
          'c1': KNeighborsClassifier, 'c1_params': {'n_neighbors': 3},
          'c2': KNeighborsClassifier})

    with pytest.raises(ValueError):
        base(x_train, x_test, y_train, y_test, opt_labels, CoTraining)

    with pytest.raises(ValueError):
        base(x_train, x_test, y_train, y_test, opt_labels, CoTraining,
             {'p': 1, 'n': 3, 'k': 100, 'u': 7})

    with pytest.raises(ValueError):
        base(x_train, x_test, y_train, y_test, opt_labels, CoTraining,
             {'p': 5, 'n': 5, 'k': 100, 'u': 15})


def test_tri_training(digits_dataset_ss):
    """
    It tests the Tri-Training algorithm on the digits dataset

    :param digits_dataset_ss: the dataset we're using
    """
    x_train, x_test, y_train, y_test, opt_labels = digits_dataset_ss
    base(x_train, x_test, y_train, y_test, opt_labels, TriTraining,
         {'c1': KNeighborsClassifier, 'c1_params': {'n_neighbors': 3},
          'c2': KNeighborsClassifier})


def test_demo_co_learning(digits_dataset_ss):
    """
    It tests the Democratic Co-Learning algorithm on the digits dataset

    :param digits_dataset_ss: The dataset we're using
    """
    x_train, x_test, y_train, y_test, opt_labels = digits_dataset_ss
    base(x_train, x_test, y_train, y_test, opt_labels, DemocraticCoLearning)
    base(x_train, x_test, y_train, y_test, opt_labels, DemocraticCoLearning,
         {'c1': KNeighborsClassifier, 'c1_params': {'n_neighbors': 3},
          'c2': KNeighborsClassifier})


def test_density_peaks(digits_dataset_ss):
    """
    It takes the training and testing data, and the optimal labels, and runs the
    STDPNF algorithm on it

    :param digits_dataset_ss: a tuple of (x_train, x_test, y_train, y_test,
    opt_labels)
    """
    x_train, x_test, y_train, y_test, opt_labels = digits_dataset_ss
    base(x_train, x_test, y_train, y_test, opt_labels, STDPNF)


def test_density_peaks_filtering(digits_dataset_ss):
    """
    It tests that the `filtering` option works as expected in the STDPNF
    algorithm

    :param digits_dataset_ss: a fixture that returns a tuple of (x_train,
    x_test, y_train, y_test, opt_labels)
    """
    x_train, x_test, y_train, y_test, opt_labels = digits_dataset_ss
    with pytest.raises(AttributeError):
        base(x_train, x_test, y_train, y_test, opt_labels, STDPNF,
             {'filtering': True})
    base(x_train, x_test, y_train, y_test, opt_labels, STDPNF,
         {'filtering': True, 'filter_method': 'ENANE'})

    base(x_train, x_test, y_train, y_test, opt_labels, STDPNF,
         {'filtering': True, 'filter_method': ENN, 'dc': 'auto',
          'classifier': KNeighborsClassifier})


def test_different_len(digits_dataset_ss):
    """
    It tests that if the length of the input and output are different, then the
    model will raise a ValueError

    :param digits_dataset_ss: a tuple of (x, x_test, y, y_test, y_pred)
    """
    x, _, y, _, _ = digits_dataset_ss
    co = CoTraining()
    tri = TriTraining()
    demo_co = DemocraticCoLearning()
    stdpnf = STDPNF()

    models = [co, tri, demo_co, stdpnf]
    y = y[:-1]

    for model in models:
        with pytest.raises(ValueError):
            model.fit(x, y)
