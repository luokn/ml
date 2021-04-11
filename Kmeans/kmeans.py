# -*- coding: utf-8 -*-
# @Date  : 2020/5/21
# @Author: Luokun
# @Email : olooook@outlook.com

import random

import matplotlib.pyplot as plt
import numpy as np


class KMeans:
    """
    K-means clustering(K均值聚类)
    """

    def __init__(self, K: int, eps=1e-3, iterations=100):
        """
        :param K: 聚类类别数
        :param eps: 中心点最小更新量
        :param max_iter: 迭代最大次数
        """
        self.K, self.eps, self.iterations = K, eps, iterations
        self.centers = None  # 中心点

    def fit(self, X: np.ndarray):
        self.centers = X[random.sample(range(len(X)), self.K)]  # 随机选择k个点作为中心点
        for _ in range(self.iterations):  # 达到最大迭代次数iterations退出迭代
            Y = self.__call__(X)  # 更新节点类别
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

    def __call__(self, X: np.ndarray):
        Y = np.empty([len(X)], dtype=int)  # 类别
        for i, x in enumerate(X):
            Y[i] = np.linalg.norm(self.centers - x, axis=1).argmin()  # 每一点类别为最近的中心点类别
        return Y


def load_data():
    x = np.stack([
        np.random.randn(200, 2) + np.array([2, 2]),
        np.random.randn(200, 2),
        np.random.randn(200, 2) + np.array([2, -2]),
    ])
    return x


if __name__ == '__main__':
    x = load_data()

    plt.figure(figsize=[12, 6])
    plt.subplot(1, 2, 1)
    plt.title('Real')
    plt.scatter(x[0, :, 0], x[0, :, 1], color='r', marker='.')
    plt.scatter(x[1, :, 0], x[1, :, 1], color='g', marker='.')
    plt.scatter(x[2, :, 0], x[2, :, 1], color='b', marker='.')

    x = x.reshape(-1, 2)
    kmeans = KMeans(3)
    kmeans.fit(x)
    pred = kmeans(x)

    z = [x[pred == c] for c in [0, 1, 2]]
    plt.subplot(1, 2, 2)
    plt.title('Pred')
    plt.scatter(z[0][:, 0], z[0][:, 1], color='r', marker='.')
    plt.scatter(z[1][:, 0], z[1][:, 1], color='g', marker='.')
    plt.scatter(z[2][:, 0], z[2][:, 1], color='b', marker='.')
    plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], color=['r', 'g', 'b'],  marker='*', s=100)
    plt.show()
