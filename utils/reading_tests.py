#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    reading_tests.py
# @Author:      Daniel Puente Ramírez
# @Time:        25/1/22 16:01

from collections.abc import Iterable
from numpy import nanmean



class DatasetResult:
    def __init__(self, name, precision, iterations, f1, mse, acc):
        self.__name = name
        self.__precision = precision
        self.__iterations = int(iterations)
        self.__f1 = [nanmean(x) for x in f1[::self.__iterations]]
        self.__f1_values = f1
        self.__mse = [nanmean(x) for x in mse[::self.__iterations]]
        self.__mse_values = mse
        self.__acc = [nanmean(x) for x in acc[::self.__iterations]]
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
            raise ValueError('Expected an iterable with the precisions')

    @property
    def iterations(self):
        return self.__iterations

    @iterations.setter
    def iterations(self, iterations):
        self.__iterations: int = iterations

    @property
    def f1(self):
        return self.__f1

    @f1.setter
    def f1(self, f1):
        if isinstance(f1, Iterable):
            self.__f1 = [nanmean(x) for x in f1[::self.__iterations]]
        else:
            raise ValueError('Expected an iterable with f1')

    @property
    def mse(self):
        return self.__mse

    @mse.setter
    def mse(self, mse):
        if isinstance(mse, Iterable):
            self.__mse = [nanmean(x) for x in mse[::self.__iterations]]
        else:
            raise ValueError('Expected an iterable with mse')

    @property
    def acc(self):
        return self.__acc

    @acc.setter
    def acc(self, acc):
        if isinstance(acc, Iterable):
            self.__acc = [nanmean(x) for x in acc[::self.__iterations]]
        else:
            raise ValueError('Expected an iterable with acc')

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
