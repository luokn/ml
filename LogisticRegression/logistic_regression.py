# -*- coding: utf-8 -*-
# @Date  : 2020/5/21
# @Author: Luokun
# @Email : olooook@outlook.com

import matplotlib.pyplot as plt
import numpy as np


class LogisticRegression:
    """
    Logistic regression classifier(逻辑斯蒂回归分类器)
    """

    def __init__(self, input_dim: int, lr: float):
        self.weights = np.random.randn(input_dim + 1)  # 随机初始化参数
        self.lr = lr  # 学习率

    def fit(self, X: np.ndarray, Y: np.ndarray):
        x_pad = self._pad(X)  # 为X填充1作为偏置
        pred = self._sigmoid(x_pad @ self.weights)  # 计算预测值
        grad = x_pad.T @ (pred - Y) / len(pred)  # 计算梯度
        self.weights -= self.lr * grad  # 沿负梯度更新参数

    def predict(self, X: np.ndarray):
        x_pad = self._pad(X)  # 为X填充1作为偏置
        pred = self._sigmoid(x_pad @ self.weights)  # 计算预测值
        return np.where(pred > 0.5, 1, 0)  # 将(0, 1)之间分布的概率转化为{0, 1}标签

    @staticmethod
    def _pad(x):
        return np.concatenate([x, np.ones([len(x), 1])], axis=1)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))


def load_data():
    x, y = np.random.randn(2, 500, 2), np.zeros([2, 500])
    x[0] += np.array([1, -1])  # 左上方移动
    x[1] += np.array([-1, 1])  # 右下方移动
    y[1] = 1
    return x, y


def train_logistic_regression(model, x, y, batch_size, epochs):
    indices = np.arange(len(x))
    for _ in range(epochs):
        np.random.shuffle(indices)
        for i in range(batch_size, len(x) + 1, batch_size):
            model.fit(x[indices[(i - batch_size):i]], y[indices[(i - batch_size):i]])


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
    x, y = load_data()
    plot_scatter(x[0], x[1], 'Real')

    # train
    x = x.reshape(-1, 2)
    y = y.flatten()
    logistic_regression = LogisticRegression(2, lr=1e-3)
    train_logistic_regression(logistic_regression, x, y, batch_size=32, epochs=100)

    # predict
    pred = logistic_regression.predict(x)
    plot_scatter_with_line(x[pred == 0], x[pred == 1], logistic_regression.weights, 'Pred')

    # accuracy
    acc = np.sum(pred == y) / len(pred)
    print(f'Acc = {100 * acc:.2f}%')
