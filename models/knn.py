# -*- coding: utf-8 -*-
# @Date  : 2020/5/20
# @Author: Luokun
# @Email : olooook@outlook.com

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


class KNN:
    """
    K nearest neighbor classifier
    """

    def __init__(self, K: int):
        self.K, self.X, self.Y = K, None, None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.X, self.Y = X, Y

    def predict(self, X: np.ndarray):
        Y = np.zeros([len(X)], dtype=int)
        for i, x in enumerate(X):
            dist = np.linalg.norm(self.X - x, axis=1)
            topk = np.argsort(dist)[:self.K]
            counter = Counter(self.Y[topk])
            Y[i] = counter.most_common(1)[0][0]
        return Y


"""
----------------------TEST----------------------
"""


def test_knn():
    x = np.random.randn(3, 100, 2)
    x[0] += np.array([2, 2])
    x[1] += np.array([2, -2])
    y = np.zeros([3, 100])
    y[1] = 1
    y[2] = 2

    # plot the real values
    plt.scatter(x[0, :, 0], x[0, :, 1], color='r', marker='.')
    plt.scatter(x[1, :, 0], x[1, :, 1], color='g', marker='.')
    plt.scatter(x[2, :, 0], x[2, :, 1], color='b', marker='.')
    plt.title("Real")
    plt.show()

    x = x.reshape(300, 2)
    y = y.reshape(300)

    # train
    knn = KNN(3)
    knn.fit(x, y)

    pred = knn.predict(x)
    x0, x1, x2 = x[pred == 0], x[pred == 1], x[pred == 2]

    # plot prediction
    plt.scatter(x0[:, 0], x0[:, 1], color='r', marker='.')
    plt.scatter(x1[:, 0], x1[:, 1], color='g', marker='.')
    plt.scatter(x2[:, 0], x2[:, 1], color='b', marker='.')
    plt.title('Pred')
    plt.show()

    # print accuracy
    acc = np.sum(pred == y) / len(pred)
    print(f'Acc = {100 * acc:.2f}%')


if __name__ == '__main__':
    test_knn()
