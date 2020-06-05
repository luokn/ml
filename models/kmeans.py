# -*- coding: utf-8 -*-
# @Date  : 2020/5/21
# @Author: Luokun
# @Email : olooook@outlook.com

import random

import numpy as np


class KMeans:
    """
    K-means clustering(K均值聚类)
    """

    def __init__(self, k: int, eps: float = 1e-3, max_iter=100):
        """
        :param k: 聚类类别数
        :param eps: 中心点最小更新量
        :param max_iter: 迭代最大次数
        """
        self.k, self.eps, self.max_iter = k, eps, max_iter
        self.centers = None  # 中心点

    def fit(self, X: np.ndarray):
        self.centers = X[random.sample(range(len(X)), self.k)]  # 随机选择k个点作为中心点
        for _ in range(self.max_iter):  # 达到最大迭代次数iterations退出迭代
            Y = self.predict(X)  # 更新节点类别
            means = np.empty_like(self.centers)  # 各类别点的均值
            for i in range(self.k):
                if np.any(Y == i):  # 存在元素属于类别i
                    means[i] = np.mean(X[Y == i], axis=0)  # 计算类别i所有点的均值
                else:  # 不存在任何元素属于类别i
                    means[i] = X[np.random.randint(0, len(X))]  # 随机选择一个点作为类别i的均值
            # 更新中心点
            if np.max(np.abs(self.centers - means)) < self.eps:  # 中心点最大更新值小于eps
                break  # 退出迭代
            self.centers = means  # 将更新后的均值作为各类别中心点

    def predict(self, X: np.ndarray):
        Y = np.empty([len(X)], dtype=int)  # 类别
        for i, x in enumerate(X):
            Y[i] = np.linalg.norm(self.centers - x, axis=1).argmin()  # 每一点类别为最近的中心点类别
        return Y
