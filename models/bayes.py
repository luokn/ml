# -*- coding: utf-8 -*-
# @Date  : 2020/5/22
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np


class NaiveBayesClassifier:
    """
    Naive Bayes classifier(朴素贝叶斯分类器, 使用贝叶斯估计)
    """

    def __init__(self, x_classes: list, y_classes: int):
        """
        :param x_classes: X所有特征的类别数
        :param y_classes: Y的类别数
        """
        self.x_classes, self.y_classes = x_classes, y_classes
        self.n_features = len(x_classes)
        self.prior_prob = np.empty([y_classes])  # 先验概率, prior_prob[i] = P(Y = i)
        self.cond_prob = [
            np.empty([len(x_classes), f_classes]) for f_classes in x_classes
        ]  # 条件概率, cond_prob[k][i,j] = P(X_i = a_{ij} | Y = k)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        :param X:
        :param Y:
        :return:
        """
        # 计算先验概率
        y_counter = np.ones([self.y_classes])  # Y类别计数器
        for y in Y:
            y_counter[y] += 1
        self.prior_prob = y_counter / y_counter.sum()  # 频率作为概率
        # 计算条件概率
        for i in range(self.y_classes):
            Xi = X[Y == i]  # 类别i的数据
            for f in range(X.shape[1]):
                x_counter = np.ones([self.x_classes[f]])  # 类别i的数据第f个特征的类别计数器
                for x in Xi[:, f]:
                    x_counter[x] += 1
                # 类别i的数据第f个特征的取值概率
                self.cond_prob[i][f] = x_counter / x_counter.sum()

    def predict(self, X: np.ndarray):
        Y = np.zeros([len(X)], dtype=int)
        for i, x in enumerate(X):
            prob = np.zeros([self.y_classes])
            for c in range(self.y_classes):
                prob[c] = np.log(self.prior_prob[c])
                prob[c] += np.log(self.cond_prob[c][:, x]).sum()
            Y[i] = prob.argmax()


if __name__ == '__main__':
    X = np.array([
        [0, 0], [0, 1], [0, 1], [0, 0], [0, 0],
        [1, 0], [1, 1], [1, 1], [1, 2], [1, 2],
        [2, 2], [2, 1], [2, 1], [2, 2], [2, 2]
    ], dtype=int)
    Y = np.array([
        0, 0, 1, 1, 0,
        0, 0, 1, 1, 1,
        1, 1, 1, 1, 0
    ], dtype=int)

    bayes = NaiveBayesClassifier([3, 3], 2)
    bayes.fit(X, Y)
    print(bayes.prior_prob)
    print(bayes.cond_prob)
