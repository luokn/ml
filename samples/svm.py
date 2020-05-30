# -*- coding: utf-8 -*-
# @Date  : 2020/5/27
# @Author: Luokun
# @Email : olooook@outlook.com

import sys
from os.path import dirname, abspath

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(dirname(dirname(abspath(__file__))))


def test_svm():
    from models.svm import SVM

    x, y = np.random.randn(2, 400, 2), np.zeros([2, 400], dtype=int)
    y[0] = -1
    y[1] = 1
    for i, theta in enumerate(np.linspace(0, 2 * np.pi, 40)):
        x[0, (10 * i):(10 * i + 10)] += 5 * np.array([np.cos(theta), np.sin(theta)])

    x = x.reshape(-1, 2)
    y = y.flatten()

    plot_scatter([x[y == i] for i in [-1, 1]], 'Real')

    # train
    svm = SVM(C=10, sigma=1, kernel='rbf', max_iter=100)
    svm.fit(x, y)

    pred = np.array(svm.predict(x))
    plot_scatter([x[pred == i] for i in [-1, 1]], 'Pred')
    acc = np.sum(pred == y) / len(pred)
    print(f'Acc = {100 * acc:.2f}%')
    print(svm.support_vectors)


def plot_scatter(xys, title):
    plt.figure(figsize=(8, 8))
    for xy, color in zip(xys, ['r', 'g', 'b']):
        plt.scatter(xy[:, 0], xy[:, 1], color=color)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    test_svm()
