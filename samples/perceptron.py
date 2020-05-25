# -*- coding: utf-8 -*-
# @Date  : 2020/5/25
# @Author: Luokun
# @Email : olooook@outlook.com
import numpy as np
import matplotlib.pyplot as plt

from models.perceptron import PerceptronClassifier


def test_perceptron():
    x, y = np.random.randn(2, 500, 2), np.zeros([2, 500], dtype=int)
    x[0] += np.array([1, -1])
    x[1] += np.array([-1, 1])
    y[0] = -1
    y[1] = 1
    plot_scatter(x[0], x[1], 'Real')

    x = x.reshape(-1, 2)
    y = y.flatten()

    perceptron = PerceptronClassifier(input_dim=2)

    train_perceptron(perceptron, x, y, batch_size=32, epochs=100)

    pred = perceptron.predict(x)
    plot_scatter_with_line(x[pred == -1], x[pred == 1], perceptron.weights, 'Pred')

    acc = np.sum(pred == y) / len(pred)
    print(f'Acc = {100 * acc:.2f}%')


def train_perceptron(model, x, y, batch_size, epochs):
    indices = np.arange(len(x))
    for epoch in range(epochs):
        np.random.shuffle(indices)
        shf_x, shf_y = x[indices], y[indices]
        bat_s, bat_e = 0, batch_size
        while bat_e <= len(x):
            model.fit(shf_x[bat_s:bat_e], shf_y[bat_s:bat_e])
            bat_s = bat_e
            bat_e += batch_size


def plot_scatter(xy0, xy1, title):
    plt.figure(figsize=[8, 8])
    plt.scatter(xy0[:, 0], xy0[:, 1], color='r', marker='.')
    plt.scatter(xy1[:, 0], xy1[:, 1], color='g', marker='.')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title(title)
    plt.show()


def plot_scatter_with_line(xy0, xy1, weights, title):
    plt.figure(figsize=[8, 8])
    plt.scatter(xy0[:, 0], xy0[:, 1], color='r', marker='.')
    plt.scatter(xy1[:, 0], xy1[:, 1], color='g', marker='.')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title(title)

    # plot the dividing line
    ln_x = np.linspace(-5, 5, 100)
    ln_a = - weights[0] / weights[1]
    ln_b = - weights[2] / weights[1]
    ln_y = ln_a * ln_x + ln_b
    plt.plot(ln_x, ln_y, color='b', linewidth=1)
    plt.show()


if __name__ == '__main__':
    test_perceptron()
