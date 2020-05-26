# -*- coding: utf-8 -*-
# @Date  : 2020/5/24
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np
import matplotlib.pyplot as plt

from models.kmeans import KMeans


def test_kmeans():
    x = np.random.randn(3, 200, 2)
    x[1] += np.array([2, 2])  # 右偏移2，上偏移2
    x[2] += np.array([2, -2])  # 右偏移2，下偏移2

    plot_scatter(x, 'Real')
    x = x.reshape(-1, 2)

    kmeans = KMeans(3)
    pred = kmeans.predict(x)
    centers = kmeans.centers

    plot_scatter_with_centers([x[pred == i] for i in [0, 1, 2]], centers, 'Pred')


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
    test_kmeans()
