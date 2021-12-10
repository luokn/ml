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
        self.prior_prob, self.cond_prob = None, None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        # 计算X的各个特征与标签Y有多少类别
        x_categories, y_categories = np.max(X, axis=0) + 1, np.max(Y) + 1
        # 先验概率, prior_prob[i] = P(Y = i)，标签类别的取值概率
        self.prior_prob = self._estimate_prob(Y, y_categories)
        # 条件概率, cond_prob[k][i,j] = P(X_i = a_{ij} | Y = k)，标签类别为k的条件下i特征取a_{ij}的概率
        self.cond_prob = [np.zeros([X.shape[1], n]) for n in x_categories]
        for k in range(y_categories):
            for i, n in enumerate(x_categories):  # 第i个特征有n_i个类别
                self.cond_prob[k][i] = self._estimate_prob(X[Y == k, i], n)  # 类别为k时，i特征所有取值的概率

    def __call__(self, X: np.ndarray):
        Y = np.zeros([len(X)], dtype=int)
        for i, x in enumerate(X):
            prob = np.log(self.prior_prob) + np.array(
                [np.sum(np.log(cond_prob[range(len(x)), x])) for cond_prob in self.cond_prob]
            )  # 先验概率的对数,加上条件概率的对数
            Y[i] = np.argmax(prob)
        return Y

    @staticmethod
    def _estimate_prob(x: np.ndarray, n: int):
        return (np.bincount(x, minlength=n) + 1) / (len(x) + n)  # 使用贝叶斯估计


def load_data():
    # 参照李航《统计学习方法（第一版）》第四章例4.1
    x = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 1],
            [0, 0],
            [0, 0],
            [1, 0],
            [1, 1],
            [1, 1],
            [1, 2],
            [1, 2],
            [2, 2],
            [2, 1],
            [2, 1],
            [2, 2],
            [2, 2],
        ],
        dtype=int,
    )
    y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0], dtype=int)
    return x, y


if __name__ == "__main__":
    x, y = load_data()
    naive_bayes = NaiveBayesClassifier()
    naive_bayes.fit(x, y)
    pred = naive_bayes(x)

    print(naive_bayes.prior_prob)  # 先验概率
    # [7/17, 10/17]
    print(naive_bayes.cond_prob[0])  # 条件概率
    # [[4/9, 3/9, 2/9]
    #  [4/9, 3/9, 2/9]]
    print(naive_bayes.cond_prob[1])  # 条件概率
    # [[3/12, 4/12, 5/12]
    #  [2/12, 5/12, 5/12]]
    acc = np.sum(pred == y) / len(pred)
    print(f"Accuracy = {100 * acc:.2f}%")
