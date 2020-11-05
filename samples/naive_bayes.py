# -*- coding: utf-8 -*-
# @Date  : 2020/5/24
# @Author: Luokun
# @Email : olooook@outlook.com


import numpy as np


def test_naive_bayes():
    import sys
    from os.path import dirname
    sys.path.append(dirname(dirname(__file__)))
    from models.naive_bayes import NaiveBayesClassifier

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

    print(bayes.prior_prob)  # 先验概率
    # [10/17, 7/17]

    print(bayes.cond_prob)  # 条件概率
    # [4/9, 3/9, 2/9]
    # [4/9, 3/9, 2/9]
    # [3/12, 4/12, 5/12]
    # [2/12, 5/12, 5/12]


if __name__ == "__main__":
    test_naive_bayes()
