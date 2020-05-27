# -*- coding: utf-8 -*-
# @Date  : 2020/5/26
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np

import json

from models.decision_tree import DecisionTreeClassifier


def test_decision_tree():
    X = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 1],

        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
    ])
    Y = np.array([1 if np.sum(x) >= 2 else 0 for x in X])

    dec_tree = DecisionTreeClassifier(rate=0.95)
    dec_tree.fit(X, Y)
    print(dec_tree.tree)

    print(Y)
    pred = dec_tree.predict(X)
    print(pred)
    acc = np.sum(pred == Y) / len(pred)
    print(f'Acc = {100 * acc:.2f}%')


if __name__ == '__main__':
    test_decision_tree()
