#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    dir.py
# @Author:      Daniel Puente Ramírez
# @Time:        9/2/22 19:15

import os


def check_dir(path):
    """
    If the path doesn't exist, create it

    :param path: the path to the folder where the ranks solutions will be saved
    """
    if not os.path.isdir(path):
        os.mkdir(path)
        if os.path.isdir(path):
            print("Created main folder for ranks solutions: ", path)
        else:
            print(
                f'Create manually the folder \'{path.split("/")[-1]}\' inside'
                f' {path.split("/")[:-1]} and  rerun.'
            )
            exit(1)
    else:
        print(f"Path OK - {path}")
