# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from models.lle import LLE

# %%

if __name__ == "__main__":
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
