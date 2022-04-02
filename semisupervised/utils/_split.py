#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    _split.py
# @Author:      Daniel Puente Ramírez
# @Time:        4/2/22 11:54

import numpy as np
import pandas as pd


def split(samples, y):
    """Split X, y into Labeled, Unlabeled and the y real tags.

    Arguments:
        X {DataFrame} -- samples
        y {DataFrame} -- labeles

    Returns:
        L {Numpy array} -- labeled samples
        U {Numpy array} -- unlabeled samples
        y {Numpy array} -- real labels
    """

    if isinstance(y, pd.DataFrame):
        y = y.to_numpy()

    labeled_indexes = y != (-1 or np.NaN or None)

    L = samples.iloc[labeled_indexes].to_numpy()
    U = samples.iloc[~labeled_indexes].to_numpy()
    y = y[labeled_indexes]

    assert len(L) == len(y), f"L {len(L)} != {len(y)} y"
    assert len(L) + len(U) == samples.shape[
        0], f"L {len(L)} + U {len(U)} != X {samples.shape[0]}"

    return L, U, y
