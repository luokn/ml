# -*- coding: utf-8 -*-
# @Date  : 2020/5/21
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np
from matplotlib import pyplot as plt


class LogisticRegression:
    """
    Logistic regression classifier(逻辑斯蒂回归分类器)
    """

    def __init__(self, input_dim: int, lr: float):
        """
        Args:
            input_dim (int):输入维度
            lr (float): 学习率
        """
        self.weights = np.random.randn(input_dim + 1)  # 随机初始化参数
        self.lr = lr  # 学习率

    def fit(self, X: np.ndarray, Y: np.ndarray):
        x_pad = self._pad(X)  # 为X填充1作为偏置
        pred = self._sigmoid(x_pad @ self.weights)  # 计算预测值
        grad = x_pad.T @ (pred - Y) / len(pred)  # 计算梯度
        self.weights -= self.lr * grad  # 沿负梯度更新参数

    def __call__(self, X: np.ndarray):
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
    x = np.stack([np.random.randn(500, 2) + np.array([1, -1]),
                  np.random.randn(500, 2) + np.array([-1, 1])])
    y = np.stack([np.full([500], 0), np.full([500], 1)])
    return x, y


def train_logistic_regression(model, x, y, epochs, batch_size=32):
    indices = np.arange(len(x))
    for _ in range(epochs):
        np.random.shuffle(indices)
        for i in range(batch_size, len(x) + 1, batch_size):
            model.fit(x[indices[(i - batch_size):i]], y[indices[(i - batch_size):i]])


if __name__ == '__main__':
    x, y = load_data()
    plt.figure(figsize=[12, 6])
    plt.subplot(1, 2, 1)
    plt.title('Real')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(x[0, :, 0], x[0, :, 1], color='r', marker='.')
    plt.scatter(x[1, :, 0], x[1, :, 1], color='g', marker='.')

    x, y = x.reshape(-1, 2), y.flatten()
    logistic_regression = LogisticRegression(2, lr=1e-3)
    train_logistic_regression(logistic_regression, x, y, epochs=500)
    pred = logistic_regression(x)
    acc = np.sum(pred == y) / len(pred)
    print(f'Accuracy = {100 * acc:.2f}%')

    x0, x1 = x[pred == 0], x[pred == 1]
    plt.subplot(1, 2, 2)
    plt.title('Pred')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(x0[:, 0], x0[:, 1], color='r', marker='.')
    plt.scatter(x1[:, 0], x1[:, 1], color='g', marker='.')

    w = logistic_regression.weights
    a, b = - w[0] / w[1], - w[2] / w[1]
    line_x = np.linspace(-5, 5, 100)
    line_y = a * line_x + b
    plt.plot(line_x, line_y, color='b', linewidth=1)
    plt.show()
