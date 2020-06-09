# -*- coding: utf-8 -*-
# @Date  : 2020/5/27
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np


class HMM:
    """
    Hidden Markov Model(隐马尔可夫模型，Viterbi算法)
    """

    def viterbi(self, Q, V, A, B, O, PI):
        N = len(Q)  # 可能存在的状态数量
        M = len(O)  # 观测序列的大小
        deltas = np.zeros((N, M))
        psis = np.zeros((N, M))
        I = np.zeros((1, M))
        for t in range(M):
            realT = t + 1
            indexOfO = V.index(O[t])  # 找出序列对应的索引
            for i in range(N):
                realI = i + 1
                if t == 0:
                    deltas[i][t] = PI[0][i] * B[i][indexOfO]
                    psis[i][t] = 0
                    print('delta1(%d)=pi%d * b%d(o1)=%.2f * %.2f=%.2f' %
                          (realI, realI, realI, PI[0][i], B[i][indexOfO],
                           deltas[i][t]))
                    print('psis1(%d)=0' % (realI))
                else:
                    deltas[i][t] = np.max(
                        np.multiply([delta[t - 1] for delta in deltas],
                                    [a[i] for a in A])) * B[i][indexOfO]
                    print(
                        'delta%d(%d)=max[delta%d(j)aj%d]b%d(o%d)=%.2f*%.2f=%.5f'
                        % (realT, realI, realT - 1, realI, realI, realT,
                           np.max(
                               np.multiply([delta[t - 1] for delta in deltas],
                                           [a[i] for a in A])), B[i][indexOfO],
                           deltas[i][t]))
                    psis[i][t] = np.argmax(
                        np.multiply(
                            [delta[t - 1] for delta in deltas],
                            [a[i]
                             for a in A])) + 1  # 由于其返回的是索引，因此应+1才能和正常的下标值相符合。
                    print('psis%d(%d)=argmax[delta%d(j)aj%d]=%d' %
                          (realT, realI, realT - 1, realI, psis[i][t]))
        print(deltas)
        print(psis)
        I[0][M - 1] = np.argmax([delta[M - 1] for delta in deltas
                                 ]) + 1  # 由于其返回的是索引，因此应+1才能和正常的下标值相符合。
        print('i%d=argmax[deltaT(i)]=%d' % (M, I[0][M - 1]))
        for t in range(M - 2, -1, -1):
            I[0][t] = psis[int(I[0][t + 1]) - 1][t + 1]
            print('i%d=psis%d(i%d)=%d' % (t + 1, t + 2, t + 2, I[0][t]))
        print("状态序列I：", I)
