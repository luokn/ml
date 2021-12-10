# -*- coding: utf-8 -*-
# @Date  : 2020/5/20
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA


class KNN:
    """
    K nearest neighbor classifier(K近邻分类器)
    """

    def __init__(self, k: int):
        """
        Args:
            k (int): 分类近邻数
        """
        self.k, self.X, self.Y = k, None, None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.X, self.Y = X, Y  # 训练集X与Y，类别已知

    def __call__(self, X: np.ndarray):
        Y = np.zeros([len(X)], dtype=int)  # X对应的类别
        for i, x in enumerate(X):
            dist = LA.norm(self.X - x, axis=1)  # 计算x与所有已知类别点的距离
            topk = np.argsort(dist)[:self.k]  # 取距离最小的k个点对应的索引
            Y[i] = np.bincount(self.Y[topk]).argmax()  # 取近邻点最多的类别作为x的类别
        return Y


def load_data():
    x = np.stack([np.random.randn(200, 2) + np.array([2, 2]),
                  np.random.randn(200, 2),
                  np.random.randn(200, 2) + np.array([2, -2])])
    y = np.stack([np.full([200], 0), np.full([200], 1), np.full([200], 2)])
    return x, y


if __name__ == '__main__':
    x, y = load_data()

    plt.figure(figsize=[12, 6])
    plt.subplot(1, 2, 1)
    plt.title('Truth')
    plt.scatter(x[0, :, 0], x[0, :, 1], color='r', marker='.')
    plt.scatter(x[1, :, 0], x[1, :, 1], color='g', marker='.')
    plt.scatter(x[2, :, 0], x[2, :, 1], color='b', marker='.')

    x, y = x.reshape(-1, 2), y.flatten()
    knn = KNN(3)
    knn.fit(x, y)
    pred = knn(x)
    acc = np.sum(pred == y) / len(pred)
    print(f'Accuracy = {100 * acc:.2f}%')

    x0, x1, x2 = x[pred == 0], x[pred == 1], x[pred == 2]
    plt.subplot(1, 2, 2)
    plt.title('Prediction')
    plt.scatter(x0[:, 0], x0[:, 1], color='r', marker='.')
    plt.scatter(x1[:, 0], x1[:, 1], color='g', marker='.')
    plt.scatter(x2[:, 0], x2[:, 1], color='b', marker='.')
    plt.show()
