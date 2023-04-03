import numpy as np
import scipy as sc 

import jax
import jax.numpy as jp
from jax.config import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

def meff(d, r, max_iter=1000, tol=1e-16,\
                             options={"disp": True,\
                                      "maxiter": 100000},\
                             method="SLSQP"):
    n = d**2
    G_MEFF = np.array([[r if i == j else r*(d*r -1)/(d**2 - 1) for j in range(n)] for i in range(n)])

    @jax.jit
    def objective(V):
        U = (V[:n*d*r] + 1j*V[n*d*r:]).reshape(n, d, r)
        G = jp.einsum("aij, akj, bkl, bil -> ab", U.conj(), U, U.conj(), U).real
        return jp.linalg.norm(G - G_MEFF) + jp.linalg.norm(jp.einsum("...ij,...ij", U.conj(), U) - jp.diag(G))

    for t in range(max_iter):
        result = sc.optimize.minimize(\
                    objective, np.random.randn(2*n*d*r),\
                    jac=jax.jit(jax.jacrev(objective)),\
                    tol=tol,\
                    options=options,\
                    method=method)
        if not np.isclose(result.fun, float("nan"), equal_nan=True):           
            U = (result.x[:n*d*r] + 1j*result.x[n*d*r:]).reshape(n, d, r)
            R = jp.einsum("...ij,...kj", U.conj(), U)
            return R

def check_meff_wh_cov(M, r, max_iter=1000, tol=1e-16, 
                                options={"disp": False,\
                                         "maxiter": 5000},\
                                method="SLSQP",\
                                return_goodies=True):
    n = M.shape[0]
    d = M.shape[1]
    R = M.reshape(n, n)
    D = displacement_operators(d)

    G_MEFF = np.array([[r if i == j else r*(d*r -1)/(d**2 - 1) for j in range(n)] for i in range(n)])
    g_meff = np.sort(G_MEFF.flatten())

    @jax.jit
    def meff_obj(V):
        U = jp.linalg.qr((V[:d**2] + 1j*V[d**2:2*d**2]).reshape(d,d), mode="complete")[0]
        X = jp.linalg.qr((V[2*d**2:2*d**2 + d*r] + 1j*V[2*d**2 + d*r:]).reshape(d, r))[0]
        S = (U @ D @ X @ X.conj().T @ np.transpose(D, (0,2,1)).conj() @ U.conj().T).reshape(n,n)
        return jp.linalg.norm(jp.sort((R @ S.conj().T).flatten()) - g_meff) 
    
    for t in range(max_iter):
        result = sc.optimize.minimize(\
                        meff_obj, np.random.randn(2*d**2 + 2*d*r),\
                        jac=jax.jit(jax.jacrev(meff_obj)),\
                        method="SLSQP")
        if np.isclose(result.fun, 0, atol=1e-3):
            if return_goodies:
                U = jp.linalg.qr((result.x[:d**2] + 1j*result.x[d**2:2*d**2]).reshape(d,d), mode='complete')[0]
                X = jp.linalg.qr((result.x[2*d**2:2*d**2 + d*r] + 1j*result.x[2*d**2 + d*r:]).reshape(d, r))[0]
                return [U, X @ X.conj().T]
            else:
                return True
    return False