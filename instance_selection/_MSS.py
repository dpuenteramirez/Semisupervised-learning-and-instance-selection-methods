#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    MSS.py
# @Author:      Daniel Puente Ramírez
# @Time:        23/11/21 11:53
# @Version:     3.0

import sys

import numpy as np
import pandas as pd

from .utils import transform


class MSS:
    """
    Barandela, R., Ferri, F. J., & Sánchez, J. S. (2005). Decision boundary
    preserving prototype selection for nearest neighbor classification.
    International Journal of Pattern Recognition and Artificial
    Intelligence, 19(06), 787-806.

    Parameters
    ----------

    """

    def __init__(self):
        """A constructor for the class."""
        self.x_attr = None

    def filter(self, samples, y):
        """
        Implementation of Modified Selective Subset.

        It starts with two empty arrays *dat* and *tar*, which will contain the
        instances selected. The first approach is to sort based on Dj all the
        dataset. And after that it will perform for every xi, check if xj is in
        X, if so it will check d(xi,xj) < Dj, if so will mark xj for deletion.
        This deletion process is not necessary due to *remove_indexes* which
        keeps track of all indexes marked for removal. If some removed had to be
        done, the *add* boolean will be set to True and it will result in xi
        being added to the results dataset.
        :param samples: DataFrame.
        :param y: DataFrame.
        :return: the input dataset with the remaining samples.
        """
        self.x_attr = samples.keys()
        samples = transform(samples, y)
        triplets = self._enemy_distance(samples["data"], samples["target"])
        dat = []
        tar = []
        remove_indexes = []

        for index, (sample, x_class, distance) in enumerate(triplets):
            if index in remove_indexes:
                continue
            add = False
            for index1 in range(index, len(triplets)):
                if index1 in remove_indexes:
                    continue
                if np.linalg.norm(sample - triplets[index1][0]) < distance:
                    remove_indexes.append(index1)
                    add = True

            if add:
                dat.append(sample)
                tar.append(x_class)

        samples = pd.DataFrame(dat, columns=self.x_attr)
        y = pd.DataFrame(tar)

        return samples, y

    @staticmethod
    def _enemy_distance(dat, tar):
        """
        For each sample in the dataset, find the distance to the nearest sample
        of a different class.

        :param dat: the data
        :param tar: the target variable
        :return: A list of lists, where each list contains a sample, its class,
         and its distance to its nearest enemy.
        """
        solution = []
        for sample, x_class in zip(dat, tar):
            distance = sys.maxsize
            for sample_1, x1_class in zip(dat, tar):
                if x1_class == x_class:
                    continue
                euc = np.linalg.norm(sample - sample_1)
                if euc < distance:
                    distance = euc
            solution.append([sample, x_class, distance])

        solution.sort(key=lambda x: x[2])

        return solution
