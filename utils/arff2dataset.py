#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    arff2dataset.py
# @Author:      Daniel Puente Ramírez
# @Time:        22/12/21 18:05
# @Version:     4.0

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch


def arff_data(dataset_path, attr=False):
    """
    It reads the arff file and returns a Bunch object with the data, target, and
    attributes

    :param dataset_path: The path to the dataset
    :param attr: If True, the dataset will return the attributes of the dataset,
    defaults to False (optional)
    :return: A bunch object with the data, target and attributes.
    """
    if ".arff" not in str(dataset_path).lower():
        raise ValueError("File does not an ARFF extension.")
    file = open(dataset_path, "r")
    attrs, data = _read_file(file)
    file.close()
    data = np.array(data)
    labels = data[:, -1]
    data = np.delete(data, -1, 1)
    data = data.astype(float)
    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)

    if not attr:
        return Bunch(data=data, target=labels)
    return Bunch(data=data, target=labels, attr=attrs)


def _read_file(file):
    """
    It reads the file and returns the attributes and data

    :param file: The file to read from
    :return: the attributes and the data.
    """
    data = []
    attrs = []
    start = False
    while True:
        next_line = file.readline()
        if not next_line:
            break
        if next_line[0] == "%":
            continue
        if "@attribute" in next_line.strip().lower():
            n = next_line.strip().split(" ")
            attrs.append(n[1] if "\t" not in n[1] else n[1].split(sep="\t")[0])
            continue

        if "@DATA" in next_line.strip().upper():
            start = True
            continue
        if start:
            line_data = next_line.strip().split(sep=",")
            data.append(np.array(line_data))
    return attrs, data
