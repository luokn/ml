# -*- coding: utf-8 -*-
# @Date  : 2020/5/21
# @Author: Luokun
# @Email : olooook@outlook.com

import random

import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA


class KMeans:
    """
    K-means clustering(K均值聚类)
    """

    def __init__(self, k: int):
        """
        Args:
            k (int): 聚类类别数
        """
        self.k, self.centers = k, None

    def fit(self, X: np.ndarray, iterations=100, eps=1e-3):
        """
        Args:
            X (np.ndarray): 输入
            iterations (int, optional): 迭代最大次数. Defaults to 100.
            eps (float, optional): 中心点最小更新量. Defaults to 1e-3.
        """
        self.centers = X[random.sample(range(len(X)), self.k)]  # 随机选择k个点作为中心点
        for _ in range(iterations):  # 达到最大迭代次数iterations退出迭代
            pred = self(X)  # 更新节点类别
            means = np.stack(
                [
                    # 存在元素属于类别i则计算类别i所有点的均值，否则随机选择一个点作为类别i的均值
                    np.mean(X[pred == i], axis=0) if np.any(pred == i) else random.choice(X)
                    for i in range(self.k)
                ]
            )  # 各类别的均值
            if np.abs(self.centers - means).max() < eps:  # 中心点最大更新值小于eps
                break  # 退出迭代
            self.centers = means  # 将更新后的均值作为各类别中心点

    def __call__(self, X: np.ndarray):
        return np.array([np.argmin(LA.norm(self.centers - x, axis=1)) for x in X])  # 每一点类别为最近的中心点类别


def load_data():
    x = np.stack(
        [
            np.random.randn(200, 2) + np.array([2, 2]),
            np.random.randn(200, 2),
            np.random.randn(200, 2) + np.array([2, -2]),
        ]
    )
    return x


if __name__ == "__main__":
    x = load_data()
    plt.figure(figsize=[12, 6])
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.scatter(x[0, :, 0], x[0, :, 1], color="r", marker=".")
    plt.scatter(x[1, :, 0], x[1, :, 1], color="g", marker=".")
    plt.scatter(x[2, :, 0], x[2, :, 1], color="b", marker=".")

    x = x.reshape(-1, 2)
    kmeans = KMeans(3)
    kmeans.fit(x)
    pred = kmeans(x)

    x0, x1, x2 = x[pred == 0], x[pred == 1], x[pred == 2]
    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.scatter(x0[:, 0], x0[:, 1], color="r", marker=".")
    plt.scatter(x1[:, 0], x1[:, 1], color="g", marker=".")
    plt.scatter(x2[:, 0], x2[:, 1], color="b", marker=".")
    plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], color=["r", "g", "b"], marker="*", s=100)
    plt.show()
