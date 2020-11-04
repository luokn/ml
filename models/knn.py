# -*- coding: utf-8 -*-
# @Date  : 2020/5/20
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np


class KNN:
    """
    K nearest neighbor classifier(K近邻分类器)
    """

    def __init__(self, k: int):
        """
        :param K: 分类近邻数
        """
        self.k = k
        self._X, self._Y = None, None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self._X, self._Y = X, Y  # 训练集X与Y，类别已知

    def predict(self, X: np.ndarray):
        Y = np.zeros([len(X)], dtype=int)  # X对应的类别输出变量
        for i, x in enumerate(X):
            dist = np.linalg.norm(self._X - x, axis=1)  # 计算x与所有已知类别点的距离
            topk = np.argsort(dist)[:self.k]  # 取距离最近的k个点对应的索引
            counter = np.bincount(self._Y[topk])  # 统计k近邻点的类别数量
            Y[i] = np.argmax(counter)  # k近邻次数最多的类别将作为x的类别
        return Y
