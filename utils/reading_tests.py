#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    reading_tests.py
# @Author:      Daniel Puente Ramírez
# @Time:        25/1/22 16:01

from collections.abc import Iterable

from numpy import nanmean


class DatasetResult:
    def __init__(self, name, precision, folds, n_samples, f1, mse, acc):
        self.__name = name
        self.__precision = precision
        self.__folds = int(folds)
        self.__n_samples = [nanmean(x) for x in n_samples[:: self.__folds]]
        self.__n_samples_values = n_samples
        self.__f1 = [nanmean(x) for x in f1[:: self.__folds]]
        self.__f1_values = f1
        self.__mse = [nanmean(x) for x in mse[:: self.__folds]]
        self.__mse_values = mse
        self.__acc = [nanmean(x) for x in acc[:: self.__folds]]
        self.__acc_values = acc

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name: str = name

    @property
    def precision(self):
        return self.__precision

    @precision.setter
    def precision(self, precision):
        if isinstance(precision, Iterable):
            self.__precision = precision
        else:
            raise ValueError("Expected an iterable with the precisions")

    @property
    def folds(self):
        return self.__folds

    @folds.setter
    def folds(self, folds):
        self.__folds: int = folds

    @property
    def n_samples(self):
        return self.__n_samples

    @n_samples.setter
    def n_samples(self, n_samples):
        if isinstance(n_samples, Iterable):
            self.__n_samples = [nanmean(x) for x in n_samples[:: self.__folds]]
        else:
            raise ValueError("Expected an iterable with the number of samples")

    @property
    def f1(self):
        return self.__f1

    @f1.setter
    def f1(self, f1):
        if isinstance(f1, Iterable):
            self.__f1 = [nanmean(x) for x in f1[:: self.__folds]]
        else:
            raise ValueError("Expected an iterable with f1")

    @property
    def mse(self):
        return self.__mse

    @mse.setter
    def mse(self, mse):
        if isinstance(mse, Iterable):
            self.__mse = [nanmean(x) for x in mse[:: self.__folds]]
        else:
            raise ValueError("Expected an iterable with mse")

    @property
    def acc(self):
        return self.__acc

    @acc.setter
    def acc(self, acc):
        if isinstance(acc, Iterable):
            self.__acc = [nanmean(x) for x in acc[:: self.__folds]]
        else:
            raise ValueError("Expected an iterable with acc")

    @property
    def f1_values(self):
        return self.__f1_values

    @f1_values.setter
    def f1_values(self, f1):
        self.__f1_values = f1

    @property
    def mse_values(self):
        return self.__mse_values

    @mse_values.setter
    def mse_values(self, mse):
        self.__mse_values = mse

    @property
    def acc_values(self):
        return self.__acc_values

    @acc_values.setter
    def acc_values(self, acc):
        self.__acc_values = acc

    @property
    def n_samples_values(self):
        return self.__n_samples_values

    @n_samples_values.setter
    def n_samples_values(self, n_samples):
        self.__n_samples_values = n_samples

    def __eq__(self, other):
        return isinstance(other, DatasetResult) and (
            self.name() == other.name() and self.precision() == other.precision()
        )

    def __hash__(self):
        return hash([self.name(), self.precision(), self.f1_values()])
