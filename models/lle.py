# -*- coding: utf-8 -*-
# @Date  : 2020/5/27
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np


class LLE:
    def __init__(self, k: int):
        self.k = k
        self.W = None

    def fit(self, X: np.array):
        self.W = np.zeros([len(X), self.k])  # N × k
        pass
