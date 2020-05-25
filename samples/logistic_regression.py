# -*- coding: utf-8 -*-
# @Date  : 2020/5/24
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np
import matplotlib.pyplot as plt

from models.logistic_regression import LogisticRegression


def test_logistic_regression():
    x = np.random.randn(2, 200, 2)
    x[0] += np.array([1, -1])
    x[1] += np.array([-1, 1])
    y = np.zeros([2, 200])
    y[1] = 1

    # plot real values
    plt.scatter(x[0, :, 0], x[0, :, 1], color='r', marker='.')
    plt.scatter(x[1, :, 0], x[1, :, 1], color='g', marker='.')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title("Real")
    plt.show()

    # prepare data
    x = x.reshape(400, 2)
    y = y.reshape(400)

    # train model
    lg_reg = LogisticRegression(2)
    train_logistic_regression(lg_reg, x, y, 32, 100)

    pred = lg_reg.predict(x)
    x0, x1 = x[pred == 0], x[pred == 1]

    # plot prediction
    plt.scatter(x0[:, 0], x0[:, 1], color='r', marker='.')
    plt.scatter(x1[:, 0], x1[:, 1], color='g', marker='.')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title("Pred")

    # plot the dividing line
    ln_x = np.linspace(x[:, 0].min(), x[:, 0].max(), 200)
    ln_a = - lg_reg.weights[0] / lg_reg.weights[1]
    ln_b = - lg_reg.weights[2] / lg_reg.weights[1]
    ln_y = ln_a * ln_x + ln_b
    plt.plot(ln_x, ln_y, color='b', linewidth=1)
    plt.show()

    acc = np.sum(pred == y) / len(pred)
    print(f'Acc = {100 * acc:.2f}%')


def train_logistic_regression(model, x, y, batch_size, epochs):
    x_len, indices = len(x), np.arange(len(x))
    for epoch in range(epochs):
        np.random.shuffle(indices)
        shf_x, shf_y = x[indices], y[indices]
        b_start, b_end = 0, batch_size
        while b_end <= x_len:
            batch_x, batch_y = shf_x[b_start:b_end], shf_y[b_start:b_end]
            model.fit(batch_x, batch_y)
            b_start = b_end
            b_end += batch_size


if __name__ == '__main__':
    test_logistic_regression()
