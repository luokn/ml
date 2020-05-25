# -*- coding: utf-8 -*-
# @Date  : 2020/5/24
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np
from models.bayes import NaiveBayesClassifier


def test_bayes_classifier():
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
    # print(bayes.predict([1, 1]))


if __name__ == "__main__":
    test_bayes_classifier()
