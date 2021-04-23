# -*- coding: utf-8 -*-
# @Date  : 2020/5/22
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np


class NaiveBayesClassifier:
    """
    Naive Bayes classifier(朴素贝叶斯分类器)
    """

    def __init__(self):
        self.x_categories, self.y_categories = None, None
        self.prior_prob, self.cond_prob = None, None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.x_categories, self.y_categories = [len(np.unique(x)) for x in X.T], len(np.unique(Y))
        # 先验概率, prior_prob[i] = P(Y = i)，表示所有标签类别取值概率
        self.prior_prob = np.empty([self.y_categories])
        # 条件概率, cond_prob[k][i,j] = P(X_i = a_{ij} | Y = k)，表示类别为K的条件下i特征取a_{ij}的概率
        self.cond_prob = [np.empty([X.shape[1], c]) for c in self.x_categories]
        # 计算先验概率
        self.prior_prob = self._estimate_prob(Y, self.y_categories)  # 频率作为概率(贝叶斯估计)
        # 计算条件概率
        for i in range(self.y_categories):
            x = X[Y == i]  # 类别i的所有数据
            for f, c in enumerate(self.x_categories):  # 第f个特征有f_classes个类别
                # 类别为i时，f特征所有取值的条件概率(贝叶斯估计)
                self.cond_prob[i][f] = self._estimate_prob(x[:, f], c)

    def __call__(self, X: np.ndarray):
        Y = np.zeros([len(X)], dtype=int)
        for i, x in enumerate(X):
            prob = np.zeros([self.y_categories])  # 每一个类别的概率
            for c in range(self.y_categories):  # 计算x为类别c的概率
                prob[c] = np.log(self.prior_prob[c])  # 先验概率(使用对数加法替代乘法避免浮点下溢)
                prob[c] += np.log(self.cond_prob[c][:, x]).sum()  # 连乘条件概率
            Y[i] = prob.argmax()
        return Y

    @staticmethod
    def _estimate_prob(x, n_classes):  # 使用贝叶斯估计
        counter = np.bincount(x, minlength=n_classes) + 1
        return counter / counter.sum()


def load_data():
    # 参照李航《统计学习方法（第一版）》第四章例4.1
    x = np.array([[0, 0], [0, 1], [0, 1], [0, 0], [0, 0],
                  [1, 0], [1, 1], [1, 1], [1, 2], [1, 2],
                  [2, 2], [2, 1], [2, 1], [2, 2], [2, 2]], dtype=int)
    y = np.array([0, 0, 1, 1, 0,
                  0, 0, 1, 1, 1,
                  1, 1, 1, 1, 0], dtype=int)
    return x, y


if __name__ == "__main__":
    x, y = load_data()
    naive_bayes = NaiveBayesClassifier()
    naive_bayes.fit(x, y)
    pred = naive_bayes(x)

    print(naive_bayes.prior_prob)  # 先验概率
    # [7/17, 10/17]
    print(naive_bayes.cond_prob)  # 条件概率
    # [4/9, 3/9, 2/9]
    # [4/9, 3/9, 2/9]
    # [3/12, 4/12, 5/12]
    # [2/12, 5/12, 5/12]
    acc = np.sum(pred == y) / len(pred)
    print(f'Accuracy = {100 * acc:.2f}%')
