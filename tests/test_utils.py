#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    utils.py
# @Author:      Daniel Puente Ramírez
# @Time:        16/4/22 16:24

from os.path import join

import pytest
from sklearn.utils import Bunch

from utils import arff_data


@pytest.fixture
def arff_path_file():
    """
    It returns the path to the iris dataset in the datasets folder
    :return: The path to the iris.arff file
    """
    return join("datasets", "iris.arff")


def test_arff_data(arff_path_file):
    """
    `arff_data` loads an arff file into a `Bunch` object, which is a
    dictionary-like object.

    :param arff_path_file: The path to the arff file
    """
    dataset = arff_data(arff_path_file)
    assert isinstance(dataset, Bunch)
    dataset1 = arff_data(arff_path_file, ["a", "b", "c", "d"])
    assert isinstance(dataset1, Bunch)
