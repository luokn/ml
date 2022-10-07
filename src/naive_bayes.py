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
        self.P_prior, self.P_cond = None, None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # 计算X的各个特征与标签Y有多少类别
        x_categories, y_categories = np.max(X, axis=0) + 1, np.max(y) + 1
        # 先验概率, prior_prob[i] = P(Y = i)，标签类别的取值概率
        self.P_prior = self.estimate_prob(y, y_categories)
        # 条件概率, cond_prob[k][i,j] = P(X_i = a_{ij} | Y = k)，标签类别为k的条件下i特征取a_{ij}的概率
        self.P_cond = [np.zeros([X.shape[1], n]) for n in x_categories]
        for k in range(y_categories):
            for i, n in enumerate(x_categories):  # 第i个特征有n_i个类别
                self.P_cond[k][i] = self.estimate_prob(X[y == k, i], n)  # 类别为k时，i特征所有取值的概率

    def __call__(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.zeros([len(X)], dtype=int)
        for i, x in enumerate(X):
            # 先验概率的对数,加上条件概率的对数
            P = np.log(self.P_prior) + np.array([np.log(p_cond[np.arange(len(x)), x]).sum() for p_cond in self.P_cond])
            y_pred[i] = np.argmax(P)
        return y_pred

    def predict_prob(self, X: np.ndarray):
        P = np.zeros([len(X), len(self.P_cond)])
        for i, x in enumerate(X):
            for cat, p_prior, p_cond in zip(range(len(self.P_cond)), self.P_prior, self.P_cond):
                P[i, cat] = p_prior * np.prod(p_cond[range(len(x)), x])
        return P

    @staticmethod
    def estimate_prob(x: np.ndarray, n: int):
        return (np.bincount(x, minlength=n) + 1) / (len(x) + n)  # 使用贝叶斯估计


def load_data():
    # 参照李航《统计学习方法（第一版）》第四章例4.1
    X = np.array(
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
    return X, y


if __name__ == "__main__":
    X, y = load_data()
    naive_bayes = NaiveBayesClassifier()
    naive_bayes.fit(X, y)
    pred = naive_bayes(X)

    print(naive_bayes.P_prior)  # 先验概率
    # [7/17, 10/17]
    print(naive_bayes.P_cond[0])  # 条件概率
    # [[4/9, 3/9, 2/9]
    #  [4/9, 3/9, 2/9]]
    print(naive_bayes.P_cond[1])  # 条件概率
    # [[3/12, 4/12, 5/12]
    #  [2/12, 5/12, 5/12]]
    acc = np.sum(pred == y) / len(pred)
    print(f"Accuracy = {100 * acc:.2f}%")
    print(naive_bayes.predict_prob([[1, 0]]))  # 输出 [[1, 0]]的概率
    # [[0.061, 0.0327]]
