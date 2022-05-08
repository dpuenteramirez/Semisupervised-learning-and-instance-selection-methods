#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    __init__.py.py
# @Author:      Daniel Puente Ramírez
# @Time:        22/12/21 11:27
"""
Semi-Supervised.

The package contains some of the most widely used semi-supervised algorithms in
the literature.
"""

__version__ = "0.1.3"
__author__ = "Daniel Puente Ramírez"

from .CoTraining import CoTraining
from .DemocraticCoLearning import DemocraticCoLearning
from .DensityPeaks import STDPNF
from .TriTraining import TriTraining

__all__ = ["CoTraining", "TriTraining", "DemocraticCoLearning", "STDPNF"]
