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


def test_knn():
    x, y = np.random.randn(3, 200, 2), np.zeros([3, 200])
    x[0] += np.array([2, 2])  # 右偏移2，上偏移2
    x[1] += np.array([2, -2])  # 右偏移2，下偏移2
    y[1] = 1
    y[2] = 2
    plot_scatter(x, 'Real')

    x = x.reshape(-1, 2)
    y = y.flatten()

    # train
    knn = KNN(3)
    knn.fit(x, y)

    pred = knn.predict(x)
    plot_scatter([x[pred == i] for i in [0, 1, 2]], 'Pred')

    # print accuracy
    acc = np.sum(pred == y) / len(pred)
    print(f'Acc = {100 * acc:.2f}%')


def plot_scatter(xys, title):
    plt.figure(figsize=(10, 10))
    for xy, color in zip(xys, ['r', 'g', 'b']):
        plt.scatter(xy[:, 0], xy[:, 1], color=color)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    test_knn()
