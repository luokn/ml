# -*- coding: utf-8 -*-
# @Date  : 2020/5/22
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np


class NaiveBayesClassifier:
    """
    Naive Bayes classifier(朴素贝叶斯分类器)
    """

    def __init__(self, x_classes: list, y_classes: int):
        """
        :param x_classes: X所有特征的类别数
        :param y_classes: Y的类别数
        """
        self.x_classes, self.y_classes = x_classes, y_classes
        self.prior_prob = np.empty([y_classes])  # 先验概率, prior_prob[i] = P(Y = i)，表示所有标签类别取值概率
        self.cond_prob = [
            np.empty([len(x_classes), classes]) for classes in x_classes
        ]  # 条件概率, cond_prob[k][i,j] = P(X_i = a_{ij} | Y = k)，表示类别为K的条件下i特征取a_{ij}的概率

    def fit(self, X: np.ndarray, Y: np.ndarray):
        # 计算先验概率
        self.prior_prob = self._estimate_prob(Y, self.y_classes)  # 频率作为概率(贝叶斯估计)
        # 计算条件概率
        for i in range(self.y_classes):
            x_i = X[Y == i]  # 类别i的所有数据
            for f, f_classes in enumerate(self.x_classes):  # 第f个特征有f_classes个类别
                # 类别为i时，f特征所有取值的条件概率(贝叶斯估计)
                self.cond_prob[i][f] = self._estimate_prob(x_i[:, f], f_classes)

    def predict(self, X: np.ndarray):
        Y = np.zeros([len(X)], dtype=int)
        for i, x in enumerate(X):
            prob = np.zeros([self.y_classes])  # 每一个类别的概率
            for c in range(self.y_classes):  # 计算x为类别c的概率
                prob[c] = np.log(self.prior_prob[c])  # 先验概率(使用对数加法替代乘法避免浮点下溢)
                prob[c] += np.log(self.cond_prob[c][:, x]).sum()  # 连乘条件概率
            Y[i] = prob.argmax()
        return Y

    @staticmethod
    def _estimate_prob(x, n_classes):  # 使用贝叶斯估计
        counter = np.bincount(x, minlength=n_classes) + 1
        return counter / counter.sum()
