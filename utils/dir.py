#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    dir.py
# @Author:      Daniel Puente Ramírez
# @Time:        9/2/22 19:15

import os


def check_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        if os.path.isdir(path):
            print(f"Created main folder for ranks solutions." f" {path}")
        else:
            print(
                f'Create manually the folder \'{path.split("/")[-1]}\' inside'
                f' {path.split("/")[:-1]} and  rerun.'
            )
            exit(1)
    else:
        print(f"Path OK - {path}")
