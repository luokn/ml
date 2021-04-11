# -*- coding: utf-8 -*-
# @Date  : 2020/5/20
# @Author: Luokun
# @Email : olooook@outlook.com

import matplotlib.pyplot as plt
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
        self.X, self.Y = None, None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.X, self.Y = X, Y  # 训练集X与Y，类别已知

    def __call__(self, X: np.ndarray):
        Y = np.zeros([len(X)], dtype=int)  # X对应的类别输出变量
        for i, x in enumerate(X):
            dist = np.linalg.norm(self.X - x, axis=1)  # 计算x与所有已知类别点的距离
            topk = np.argsort(dist)[:self.k]  # 取距离最近的k个点对应的索引
            counter = np.bincount(self.Y[topk])  # 统计k近邻点的类别数量
            Y[i] = np.argmax(counter)  # k近邻次数最多的类别将作为x的类别
        return Y


def load_data():
    x = np.stack([
        np.random.randn(200, 2) + np.array([2, 2]),
        np.random.randn(200, 2),
        np.random.randn(200, 2) + np.array([2, -2]),
    ])
    y = np.stack([np.full([200], 0), np.full([200], 1), np.full([200], 2)])
    return x, y


def plot_scatter(xys, title):
    plt.figure(figsize=(10, 10))
    for xy, color in zip(xys, ['r', 'g', 'b']):
        plt.scatter(xy[:, 0], xy[:, 1], color=color)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    x, y = load_data()

    plt.figure(figsize=[12, 6])
    plt.subplot(1, 2, 1)
    plt.title('Real')
    plt.scatter(x[0, :, 0], x[0, :, 1], color='r', marker='.')
    plt.scatter(x[1, :, 0], x[1, :, 1], color='g', marker='.')
    plt.scatter(x[2, :, 0], x[2, :, 1], color='b', marker='.')

    x, y = x.reshape(-1, 2), y.flatten()
    knn = KNN(3)
    knn.fit(x, y)
    pred = knn(x)
    acc = np.sum(pred == y) / len(pred)
    print(f'Acc = {100 * acc:.2f}%')

    z = [x[pred == c] for c in [0, 1, 2]]
    plt.subplot(1, 2, 2)
    plt.title('Pred')
    plt.scatter(z[0][:, 0], z[0][:, 1], color='r', marker='.')
    plt.scatter(z[1][:, 0], z[1][:, 1], color='g', marker='.')
    plt.scatter(z[2][:, 0], z[2][:, 1], color='b', marker='.')
    plt.show()
