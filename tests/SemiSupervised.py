#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    SemiSupervised.py
# @Author:      Daniel Puente Ramírez
# @Time:        16/4/22 00:22

import random

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris as load_digits
from sklearn.model_selection import train_test_split

from semisupervised import STDPNF, CoTraining, TriTraining, \
    DemocraticCoLearning


def to_dataframe(y):
    if not isinstance(y, pd.DataFrame):
        return pd.DataFrame(y)
    return y


@pytest.fixture
def digits_dataset_ss():
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
    unlabeled = random.sample(li, int(x_train.shape[0] * 0.3))
    y_train.loc[unlabeled] = -1

    return x_train, x_test, y_train, y_test, opt_labels


def base(x_train, x_test, y_train, y_test, opt_labels, algorithm, params=None):
    assert isinstance(x_train, pd.DataFrame) and isinstance(y_train,
                                                            pd.DataFrame)
    assert isinstance(x_test, pd.DataFrame) and isinstance(y_test,
                                                           pd.DataFrame)
    model = algorithm(**params) if params is not None else algorithm()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    assert set(y_pred).issubset(opt_labels)


def test_co_training(digits_dataset_ss):
    x_train, x_test, y_train, y_test, opt_labels = digits_dataset_ss
    base(x_train, x_test, y_train, y_test, opt_labels, CoTraining)


def test_tri_training(digits_dataset_ss):
    x_train, x_test, y_train, y_test, opt_labels = digits_dataset_ss
    base(x_train, x_test, y_train, y_test, opt_labels, TriTraining)


def test_demo_co_learning(digits_dataset_ss):
    x_train, x_test, y_train, y_test, opt_labels = digits_dataset_ss
    base(x_train, x_test, y_train, y_test, opt_labels, DemocraticCoLearning)


def test_density_peaks(digits_dataset_ss):
    x_train, x_test, y_train, y_test, opt_labels = digits_dataset_ss
    base(x_train, x_test, y_train, y_test, opt_labels, STDPNF)
