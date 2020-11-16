import sys
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def make_swiss_roll(n_samples=100, noise=0.0, shuffle=True):
    t = 1.5 * np.pi * (1 + 2 * np.linspace(0, 1, n_samples))
    X = np.stack([
        t * np.cos(t),
        t * np.sin(t),
        100 * np.random.rand(n_samples),
    ], axis=-1) + noise * np.random.randn(n_samples, 3)  # N × (x, y, z)
    C = np.stack([
        np.linspace(0.2, 1, n_samples),
        np.linspace(1, 0.2, n_samples),
        np.repeat(.5, n_samples)
    ], axis=-1)  # N × (r, g, b)
    if shuffle:
        index = np.arange(n_samples).astype(np.int)
        np.random.shuffle(index)
        X, C = X[index], C[index]
    return X, C


if __name__ == "__main__":
    from os.path import dirname
    sys.path.append(dirname(dirname(__file__)))
    from models.lle import LLE
    n_samples = 10000
    lle = LLE(3, 1)
    X, C = make_swiss_roll(n_samples, 0.1)
    ax = Axes3D(plt.figure(figsize=(8, 8)))
    ax.scatter(*X.T, c=C, marker='.')
    plt.show()
    # Y = lle.transform(X)
    # plt.figure(figsize=(8, 8))
    # plt.scatter(Y[:, 0], np.repeat(0, n_samples), c=C)
    # plt.show()
