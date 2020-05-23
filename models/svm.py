# -*- coding: utf-8 -*-
# @Date  : 2020/5/23
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np


class SVM:
    """
    Support Vector Machines
    """

    def __init__(self, input_dim: int, C=1.0, kernel='linear'):
        self.C = C
        self.input_dim = input_dim
        self.kernel = self.kernels[kernel]
        self.weights = np.random.randn(input_dim + 1)
        self.alpha = np.random.randn(input_dim)
        self.vectors = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        pass

    def predict(self, X: np.ndarray):
        pass

    @staticmethod
    def linear_kernel(a: np.ndarray, b: np.ndarray):
        pass

    @staticmethod
    def poly_kernel(a: np.ndarray, b: np.ndarray):
        pass

    @staticmethod
    def rbf_kernel(a: np.ndarray, b: np.ndarray):
        pass

    kernels = {
        'linear': linear_kernel,
        'poly': poly_kernel,
        'rbf': rbf_kernel
    }
