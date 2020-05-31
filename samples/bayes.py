# -*- coding: utf-8 -*-
# @Date  : 2020/5/24
# @Author: Luokun
# @Email : olooook@outlook.com

import sys
from os.path import dirname, abspath

import numpy as np

sys.path.append(dirname(dirname(abspath(__file__))))


def test_naive_bayes():
    from models.bayes import NaiveBayesClassifier

    # 参照李航《统计学习方法（第一版）》第四章例4.1
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
    print(bayes.prior_prob)  # 计算先验概率
    print(bayes.cond_prob)  # 计算条件概率


if __name__ == "__main__":
    test_naive_bayes()
