#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    testing_driver.py
# @Author:      Daniel Puente Ramírez
# @Time:        5/1/22 15:54

from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from TriTraining import TriTraining

if __name__ == '__main__':
    model = TriTraining(learn=1, random_state=42)
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.4,
                                                        stratify=y,
                                                        random_state=42)
    model.fit(L=X_train, U=X_test, y=y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
