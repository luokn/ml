# -*- coding: utf-8 -*-
# @Date  : 2020/5/24
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np
import matplotlib.pyplot as plt

from models.kmeans import KMeans


def test_kmeans():
    x = np.random.randn(3, 100, 2)
    x[0] += np.array([2, 2])
    x[1] += np.array([2, -2])

    # plot real values
    plt.scatter(x[0, :, 0], x[0, :, 1], color='r', marker='.')
    plt.scatter(x[1, :, 0], x[1, :, 1], color='g', marker='.')
    plt.scatter(x[2, :, 0], x[2, :, 1], color='b', marker='.')
    plt.title("Real")
    plt.show()

    x = x.reshape(300, 2)
    kmeans = KMeans(3)
    pred = kmeans.predict(x)
    centers = kmeans.centers

    x0, c0 = x[pred == 0], centers[0]
    x1, c1 = x[pred == 1], centers[1]
    x2, c2 = x[pred == 2], centers[2]

    # plot prediction
    plt.scatter(x0[:, 0], x0[:, 1], color='r', marker='.')
    plt.scatter(c0[0], c0[1], color='r', s=100, marker='*')
    plt.scatter(x1[:, 0], x1[:, 1], color='g', marker='.')
    plt.scatter(c1[0], c1[1], color='g', s=100, marker='*')
    plt.scatter(x2[:, 0], x2[:, 1], color='b', marker='.')
    plt.scatter(c2[0], c2[1], color='b', s=100, marker='*')
    plt.title('Pred')
    plt.show()


if __name__ == '__main__':
    test_kmeans()
