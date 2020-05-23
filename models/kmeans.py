# -*- coding: utf-8 -*-
# @Date  : 2020/5/21
# @Author: Luokun
# @Email : olooook@outlook.com

import random
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    """
    K-means clustering
    """

    def __init__(self, k: int, eps: float = 1e-3, iterations=100):
        self.k, self.eps, self.iterations = k, eps, iterations
        self.centers = None

    def predict(self, X: np.ndarray):
        Y = np.zeros([len(X)], dtype=int)
        self.centers = X[random.sample(range(len(X)), self.k)]  # centers
        for _ in range(self.iterations):
            for i, x in enumerate(X):
                Y[i] = np.linalg.norm(self.centers - x, axis=1).argmax()
            means = np.empty_like(self.centers)  # means
            for i in range(self.k):
                if np.any(Y == i):
                    means[i] = np.mean(X[Y == i], axis=0)
                else:
                    means[i] = X[np.random.randint(0, len(X))]
            if np.max(np.abs(self.centers - means)) < self.eps:
                break
            self.centers = means
        return Y


"""
----------------------TEST----------------------
"""


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
