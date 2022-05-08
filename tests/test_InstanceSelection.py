#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    InstanceSelection.py
# @Author:      Daniel Puente Ramírez
# @Time:        15/4/22 16:20

import random

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from instance_selection import ENN, CNN, RNN, ICF, MSS, DROP3, LSSm, LSBo


def to_dataframe(y):
    """
    If the input is not a dataframe, convert it to a dataframe

    :param y: The target variable
    :return: A dataframe
    """
    if not isinstance(y, pd.DataFrame):
        return pd.DataFrame(y)


@pytest.fixture
def iris_dataset():
    """
    It loads the iris dataset, converts the target variable to a dataframe, and
    returns the data and target
    :return: The dataframe of the features and the dataframe of the labels
    """
    x, y = load_iris(return_X_y=True, as_frame=True)
    y = to_dataframe(y)
    return x, y


@pytest.fixture
def iris_dataset_ss():
    """
    It loads the iris dataset, randomly selects 30% of the data points to be
    unlabeled, and returns the labeled and unlabeled data
    :return: The original dataframe, the original labels, the complete
    dataframe, and the complete labels.
    """
    x, y = load_iris(return_X_y=True, as_frame=True)
    y = to_dataframe(y)
    li = list(set(range(x.shape[0])))

    unlabeled = random.sample(li, int(x.shape[0] * 0.3))
    labeled = [x for x in range(x.shape[0]) if x not in unlabeled]

    complete = x
    complete_labels = y

    original = x.loc[labeled]
    original_labels = y.loc[labeled]

    return original, original_labels, complete, complete_labels


def base(x, y, algorithm, params=None):
    """
    It takes in a dataframe of features, a dataframe of labels, an algorithm,
    and a dictionary of parameters. It then creates a model using the
    algorithm and parameters, and filters the data using the model

    :param x: The input dataframe
    :param y: The target variable
    :param algorithm: The algorithm to use
    :param params: a dictionary of parameters to pass to the algorithm
    """
    assert isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame)
    model = algorithm(**params) if params is not None else algorithm()
    x_filtered, y_filtered = model.filter(x, y)

    assert x_filtered.shape[1] == x.shape[1] and y_filtered.shape[1] == \
           y.shape[1]

    assert x_filtered.shape[0] == y_filtered.shape[0]
    assert x_filtered.shape[0] < x.shape[0]


def test_enn_original(iris_dataset):
    """
    It tests the ENN algorithm on the iris dataset

    :param iris_dataset: This is the dataset to use
    """
    x, y = iris_dataset
    base(x, y, ENN, {'nearest_neighbors': 3, 'power_parameter': 2})


def test_cnn(iris_dataset):
    """
    It tests the CNN algorithm on the iris dataset

    :param iris_dataset: This is the dataset to use
    """
    x, y = iris_dataset
    base(x, y, CNN)


def test_rnn(iris_dataset):
    """
    It tests the RNN algorithm on the iris dataset

    :param iris_dataset: This is the dataset to use
    """
    x, y = iris_dataset
    base(x, y, RNN)


def test_icf(iris_dataset):
    """
    It tests the ICF algorithm on the iris dataset

    :param iris_dataset: This is the dataset to use
    """
    x, y = iris_dataset
    base(x, y, ICF, {'nearest_neighbors': 3, 'power_parameter': 2})


def test_mss(iris_dataset):
    """
    It tests the MSS algorithm on the iris dataset

    :param iris_dataset: This is the dataset to use
    """
    x, y = iris_dataset
    base(x, y, MSS)


def test_drop3(iris_dataset):
    """
    It tests the DROP3 algorithm on the iris dataset

    :param iris_dataset: This is the dataset to use
    """
    x, y = iris_dataset
    base(x, y, DROP3, {'nearest_neighbors': 3, 'power_parameter': 2})


def test_local_sets_lssm(iris_dataset):
    """
    It tests the LSSm algorithm on the iris dataset

    :param iris_dataset: This is the dataset to use
    """
    x, y = iris_dataset
    base(x, y, LSSm)


def test_local_sets_lsbo(iris_dataset):
    """
    It tests the LSBo algorithm on the iris dataset

    :param iris_dataset: This is the dataset to use
    """
    x, y = iris_dataset
    base(x, y, LSBo)


def test_enn_ss(iris_dataset_ss):
    """
    It tests the safe ENN algorithm on the iris dataset

    :param iris_dataset: This is the dataset to use
    """
    original, original_labels, complete, complete_labels, = iris_dataset_ss

    model = ENN()
    x, y = model.filter_original_complete(original, original_labels,
                                          complete, complete_labels)

    new_orig = []
    for ori in original.to_numpy():
        for index, x_sample in enumerate(x.to_numpy()):
            if np.array_equal(ori, x_sample):
                new_orig.append(index)
                break

    a = np.ravel(y.loc[new_orig].to_numpy())
    o = np.ravel(original_labels.to_numpy())
    assert np.array_equal(o, a)
    assert complete.shape[1] == x.shape[1]
    assert complete.shape[0] >= x.shape[0]


def test_different_len(iris_dataset):
    """
    It tests that the filter method raises a ValueError if the length of the
    input and output dataframes are different

    :param iris_dataset: a fixture that returns a tuple of two pandas
    DataFrames, one for the features and one for the target
    """
    x, y = iris_dataset
    y = y.loc[:-1]
    model1 = LSSm()
    with pytest.raises(ValueError):
        model1.filter(x, y)
    model2 = LSBo()
    with pytest.raises(ValueError):
        model2.filter(x, y)
