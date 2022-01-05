# -*- coding: utf-8 -*-
# @Date  : 2020/5/20
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np
from matplotlib import pyplot as plt


class Perceptron:
    """
    Perceptron classifier(感知机分类器)
    """

    def __init__(self, input_dim: int):
        """
        Args:
            input_dim (int): 输入特征维度
        """
        self.input_dim = input_dim
        self.weights = np.random.randn(input_dim + 1)  # 权重

    def fit(self, X: np.ndarray, Y: np.ndarray, lr=1e-3):
        for x, y in zip(pad(X), Y):
            if y * (x @ self.weights) <= 0:  # 分类错误, y 与 wx + b 符号不同
                neg_grad = x * y  # 计算weights的负梯度
                self.weights += lr * neg_grad  # 沿负梯度方向更新weights

    def __call__(self, X: np.ndarray):
        pred = pad(X) @ self.weights
        return np.where(pred > 0, 1, -1)


def pad(x):
    return np.concatenate([x, np.ones([len(x), 1])], axis=1)


def load_data():
    x = np.stack([np.random.randn(500, 2) + np.array([1, -1]), np.random.randn(500, 2) + np.array([-1, 1])])
    y = np.stack([np.full([500], -1), np.full([500], 1)])
    return x, y


def train_perceptron(model, x, y, epochs):
    indices = np.arange(len(x))
    for _ in range(epochs):
        np.random.shuffle(indices)
        model.fit(x[indices], y[indices])


if __name__ == "__main__":
    x, y = load_data()
    plt.figure(figsize=[12, 6])
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(x[0, :, 0], x[0, :, 1], color="r", marker=".")
    plt.scatter(x[1, :, 0], x[1, :, 1], color="g", marker=".")

    x, y = x.reshape(-1, 2), y.flatten()
    perceptron = Perceptron(input_dim=2)
    train_perceptron(perceptron, x, y, epochs=100)
    pred = perceptron(x)
    acc = np.sum(pred == y) / len(pred)
    print(f"Accuracy = {100 * acc:.2f}%")

    x0, x1 = x[pred == -1], x[pred == 1]
    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(x0[:, 0], x0[:, 1], color="r", marker=".")
    plt.scatter(x1[:, 0], x1[:, 1], color="g", marker=".")

    w = perceptron.weights
    a, b = -w[0] / w[1], -w[2] / w[1]
    line_x = np.linspace(-5, 5, 100)
    line_y = a * line_x + b
    plt.plot(line_x, line_y, color="b", linewidth=1)
    plt.show()
