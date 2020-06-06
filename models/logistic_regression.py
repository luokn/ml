# -*- coding: utf-8 -*-
# @Date  : 2020/5/21
# @Author: Luokun
# @Email : olooook@outlook.com

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
