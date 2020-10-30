# -*- coding: utf-8 -*-
# @Date  : 2020/5/31
# @Author: Luokun
# @Email : olooook@outlook.com


import numpy as np
import matplotlib.pyplot as plt


def test_adaboost():
    from models.adaboost import AdaBoost

    x, y = np.random.randn(2, 200, 2), np.zeros([2, 200])
    x[0] += np.array([-1, 0])
    x[1] += np.array([1, -1])
    y[0] = -1
    y[1] = 1
    plot_scatter(x, 'Real')

    x = x.reshape(-1, 2)
    y = y.flatten()

    # train
    ab = AdaBoost(5)
    ab.fit(x, y)

    pred = ab.predict(x)
    plot_scatter([x[pred == i] for i in [-1, 1]], 'Pred')

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
    test_adaboost()
