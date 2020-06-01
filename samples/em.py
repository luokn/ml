# -*- coding: utf-8 -*-
# @Date  : 2020/6/1
# @Author: Luokun
# @Email : olooook@outlook.com

import sys
from os.path import dirname, abspath

import numpy as np

sys.path.append(dirname(dirname(abspath(__file__))))


def test_em_coin():
    from models.em import SimpleEM
    y = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    em = SimpleEM([.5, .5, .5], 100)
    em.fit(y)
    print(em.prob)  # [0.5, 0.6, 0.6]

    em = SimpleEM([.4, .6, .7], 100)
    em.fit(y)
    print(em.prob)  # [0.4064, 0.5368, 0.6432]


if __name__ == '__main__':
    test_em_coin()
