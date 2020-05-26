# -*- coding: utf-8 -*-
# @Date  : 2020/5/24
# @Author: Luokun
# @Email : olooook@outlook.com

import matplotlib.pyplot as plt
import numpy as np

from models.knn import KNN


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
