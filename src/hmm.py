import numpy as np
import random


class HMM:
    """
    hmm 主要可以由以下三个问题组成:
    1. 给定模型参数和观测序列, 计算该观测序列出现的概率. (前向算法)
    2. 给定模型参数和观测序列, 计算最有可能出现的状态序列. (Viterbi 算法)
    3. 根据观测序列推测最有可能出现的模型参数. (EM 算法)
    """

    def __init__(self, s_num: int, y_num: int) -> None:
        self.P = np.ones((s_num, s_num)) / s_num
        self.Q = np.ones((s_num, y_num)) / y_num
        self.pi = np.ones((s_num,)) / s_num

    def forward(self, y: np.ndarray):
        return forward(y, self.P, self.Q, self.pi)

    def backward(self, y: np.ndarray):
        return backward(y, self.P, self.Q)

    def viterbi(self, y: np.ndarray):
        return viterbi(y, self.P, self.Q, self.pi)

    def fit(self, y: np.ndarray, iter: int = 100):
        for _ in range(iter):
            self.em_step(y)

    def em_step(self, y: np.ndarray):
        alpha = self.forward(y)
        beta = self.backward(y)

        gamma = alpha / np.sum(alpha[-1]) * beta

        ksi = np.zeros((len(y) - 1, len(self.P), len(self.P)))

        for t in range(len(y) - 1):
            ksi[t] = (
                np.reshape(alpha[t], (-1, 1))
                * self.P
                * self.Q[:, int(y[t + 1])]
                * beta[t + 1]
            ) / np.sum(alpha[-1])

        self.pi = gamma[0]
        self.P = np.sum(ksi, axis=0) / np.reshape(
            np.sum(gamma, axis=0) - gamma[-1], (-1, 1)
        )

        self.Q = np.zeros_like(self.Q)
        for j in range(self.Q.shape[1]):
            self.Q[:, j] = np.sum(np.reshape(y == j, (-1, 1)) * gamma, axis=0)
        self.Q /= np.reshape(np.sum(gamma, axis=0), (-1, 1))


def forward(y: np.ndarray, P: np.ndarray, Q: np.ndarray, pi: np.ndarray):
    n = len(y)
    alpha = np.zeros((n, P.shape[1]))
    alpha[0] = pi * Q[:, int(y[0])]
    for i in range(1, n):
        alpha[i] = (
            np.sum(np.reshape(alpha[i - 1], (-1, 1)) * P, axis=0) * Q[:, int(y[i])]
        )
    return alpha


def backward(y: np.ndarray, P: np.ndarray, Q: np.ndarray):
    n = len(y)
    beta = np.ones((n, P.shape[1]))
    for i in range(-2, -n - 1, -1):
        beta[i] = np.sum(beta[i + 1] * P * Q[:, int(y[i + 1])], axis=1)
    return beta


def viterbi(y: np.ndarray, P: np.ndarray, Q: np.ndarray, pi: np.ndarray):
    n = len(y)
    A = np.zeros((n, len(pi)))
    gamma = pi * Q[:, int(y[0])]
    for i in range(1, n):
        A[i] = np.argmax(np.reshape(gamma, (-1, 1)) * P, axis=0)
        gamma = np.max(np.reshape(gamma, (-1, 1)) * P, axis=0) * Q[:, int(y[i])]
    ans = np.zeros((n,))
    ans[-1] = np.argmax(gamma)
    for i in range(-2, -n - 1, -1):
        ans[i] = A[i + 1][int(ans[i + 1])]
    return ans


def generate_data(P: np.ndarray, Q: np.ndarray, pi: np.ndarray, T: int):
    s = np.zeros((T,))
    y = np.zeros((T,))
    s[0] = random_choose(pi)
    y[0] = random_choose(Q[int(s[0])])
    for i in range(1, T):
        s[i] = random_choose(P[int(s[i - 1])])
        y[i] = random_choose(Q[int(s[i])])
    return s, y


def random_choose(probs: np.ndarray):
    x = random.random()
    prob = probs.copy()
    if x < prob[0]:
        return 0
    for i in range(1, len(prob)):
        prob[i] += prob[i - 1]
        if x < prob[i]:
            return i
    return len(prob) - 1


if __name__ == "__main__":
    P = np.array([[0.3, 0.7], [0.6, 0.4]])
    Q = np.array([[0.9, 0.1], [0.1, 0.9]])
    pi = np.array([0.5, 0.5])
    hmm = HMM(2, 2)
    hmm.P = P
    s, y = generate_data(P, Q, pi, 5)
    hmm.fit(y, 20)
    print(hmm.Q)
