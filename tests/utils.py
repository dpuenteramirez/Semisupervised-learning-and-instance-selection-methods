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
    return join('datasets', 'iris.arff')


def test_arff_data(arff_path_file):
    dataset = arff_data(arff_path_file)
    assert isinstance(dataset, Bunch)
    dataset1 = arff_data(arff_path_file, ['a', 'b', 'c', 'd'])
    assert isinstance(dataset1, Bunch)
