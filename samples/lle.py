# %%
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# %%

if __name__ == "__main__":
    import sys
    from os.path import dirname
    sys.path.append(dirname(dirname(__file__)))
    from models.lle import LLE
    lle = LLE(3, 1)
    X = np.random.randn(10, 3)
    Y = lle.transform(X)
    print(Y)

# %%

# %%
X = np.random.uniform(size=[1000, 2])
C = np.zeros([1000, 3])
C[:, 0] = X[:, 0]
C[:, 1] = 1 - X[:, 0]
C[:, 2] = 0.5
X[:, 0] *= 3 * np.pi
# %%
plt.figure(figsize=(30, 10))
plt.scatter(X[:, 0], X[:, 1], c=C)
# %%

# %%
