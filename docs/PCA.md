
Python实现：
```python
import numpy as np


def pca(X: np.ndarray, K: int):
    """
    Principal Components Analysis
    """
    Y = X - X.mean(axis=0)
    L, U = np.linalg.eig(Y.T @ Y)
    topk = np.argsort(L)[::-1][:K]
    return Y @ U[:, topk]
```


降维前：
![降维前](../images/pca/0.png)

PCA 2D：
![降维后(2D)](../images/pca/1.png)

PCA 1D：
![降维后(1D)](../images/pca/2.png)