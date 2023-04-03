# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/factorize.ipynb.

# %% auto 0
__all__ = ['full_rank_factorization']

# %% ../nbs/factorize.ipynb 4
import numpy as np

# %% ../nbs/factorize.ipynb 5
def full_rank_factorization(M):
    U, D, V = np.linalg.svd(M)
    m = (np.isclose(D, 0)).argmax()
    m = m if m != 0 else len(D)
    sqrtD = np.sqrt(np.diag(D[:m]))
    return U[:,:m] @ sqrtD, sqrtD @ V[:m,:]