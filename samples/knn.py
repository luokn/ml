# -*- coding: utf-8 -*-
# @Date  : 2020/5/24
# @Author: Luokun
# @Email : olooook@outlook.com

import matplotlib.pyplot as plt
import numpy as np

from models.knn import KNN


def test_knn():
    x, y = np.random.randn(3, 100, 2), np.zeros([3, 100])
    x[0] += np.array([2, 2])
    x[1] += np.array([2, -2])
    y[1] = 1
    y[2] = 2

    # plot the real values
    scatter(x[0], x[1], x[2], 'Real')

    x = x.reshape(-1, 2)
    y = y.flatten()

    # train
    knn = KNN(3)
    knn.fit(x, y)

    pred = knn.predict(x)

    # plot prediction
    scatter(x[pred == 0], x[pred == 1], x[pred == 2], 'Pred')

    # print accuracy
    acc = np.sum(pred == y) / len(pred)
    print(f'Acc = {100 * acc:.2f}%')


def scatter(xy0, xy1, xy2, title):
    plt.figure(figsize=(10, 10))
    plt.scatter(xy0[:, 0], xy0[:, 1], color='r', marker='.')
    plt.scatter(xy1[:, 0], xy1[:, 1], color='g', marker='.')
    plt.scatter(xy2[:, 0], xy2[:, 1], color='b', marker='.')
    plt.show()


if __name__ == '__main__':
    test_knn()
