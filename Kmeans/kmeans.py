# -*- coding: utf-8 -*-
# @Date  : 2020/5/21
# @Author: Luokun
# @Email : olooook@outlook.com

import matplotlib.pyplot as plt
import random

import numpy as np
from numpy.lib.npyio import load


class KMeans:
    """
    K-means clustering(K均值聚类)
    """

    def __init__(self, K: int, eps=1e-3, max_iter=100):
        """
        :param K: 聚类类别数
        :param eps: 中心点最小更新量
        :param max_iter: 迭代最大次数
        """
        self.K, self.eps, self.max_iter = K, eps, max_iter
        self.centers = None  # 中心点

    def fit(self, X: np.ndarray):
        self.centers = X[random.sample(range(len(X)), self.K)]  # 随机选择k个点作为中心点
        for _ in range(self.max_iter):  # 达到最大迭代次数iterations退出迭代
            Y = self.predict(X)  # 更新节点类别
            means = np.empty_like(self.centers)  # 各类别点的均值
            for k in range(self.K):
                if np.any(Y == k):  # 存在元素属于类别i
                    means[k] = np.mean(X[Y == k], axis=0)  # 计算类别i所有点的均值
                else:  # 不存在任何元素属于类别i
                    means[k] = X[np.random.randint(0, len(X))]  # 随机选择一个点作为类别i的均值
            # 更新中心点
            if np.max(np.abs(self.centers - means)) < self.eps:  # 中心点最大更新值小于eps
                break  # 退出迭代
            self.centers = means  # 将更新后的均值作为各类别中心点

    def predict(self, X: np.ndarray):
        Y = np.empty([len(X)], dtype=int)  # 类别
        for i, x in enumerate(X):
            Y[i] = np.linalg.norm(self.centers - x, axis=1).argmin()  # 每一点类别为最近的中心点类别
        return Y


def load_data():
    x = np.random.randn(3, 200, 2)
    x[1] += np.array([2, 2])  # 右偏移2，上偏移2
    x[2] += np.array([2, -2])  # 右偏移2，下偏移2
    return x


def plot_scatter(xys, title):
    plt.figure(figsize=[8, 8])
    for xy, color in zip(xys, ['r', 'g', 'b']):
        plt.scatter(xy[:, 0], xy[:, 1], color=color, marker='.')
    plt.title(title)
    plt.show()


def plot_scatter_with_centers(xys, centers, title):
    plt.figure(figsize=[8, 8])
    for xy, center, color in zip(xys, centers, ['r', 'g', 'b']):
        plt.scatter(xy[:, 0], xy[:, 1], color=color, marker='.')
        plt.scatter(center[0], center[1], color=color, s=100, marker='*')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    x = load_data()
    plot_scatter(x, 'Real')

    x = x.reshape(-1, 2)
    kmeans = KMeans(3)
    pred = kmeans.predict(x)

    plot_scatter_with_centers([x[pred == i] for i in [0, 1, 2]], kmeans.centers, 'Pred')
