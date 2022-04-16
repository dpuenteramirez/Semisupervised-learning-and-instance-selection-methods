# !/usr/bin/env python -*- coding:utf-8 -*- @Filename:    threads.py @Author:
# https://alexandra-zaharia.github.io/posts/how-to-return-a-result-from-a
# -python-thread/


import sys
import threading


class ReturnValueThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result = None

    def run(self):
        if self._target is None:
            return
        try:
            self.result = self._target(*self._args, **self._kwargs)
        except Exception as exc:
            print(f'{type(exc).__name__}: {exc}', file=sys.stderr)

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        return self.result
