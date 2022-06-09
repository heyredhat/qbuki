import scipy as sc
import numpy as np
from functools import partial

import jax
import jax.numpy as jp
from jax.config import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

from .utils import *

@partial(jax.jit, static_argnums=(1))
def jit_spectral_inverse(P, r):
    n = P.shape[0]
    a, A, B = [1], [P], []
    for i in range(1, n+1):
        a.append(A[-1].trace()/i)
        B.append(A[-1] - a[-1]*jp.eye(n))
        A.append(P @ B[-1])
    j = n - r
    return sum([((-1 if i == 0 else 1)*a[n-j-1]*a[i]/a[n-j]**2 + \
                 (i if i < 2 else -1)*a[i-1]/a[n-j])*\
                    jp.linalg.matrix_power(P, n-j-i)
                        for i in range(r)])

@partial(jax.jit, static_argnums=(1))
def jit_pnorm(A, p):
    n = A.shape[0]
    S = jp.linalg.svd(np.eye(n) - A, compute_uv=False)
    return jp.sum(S**p)**(1/p) if p != jp.inf else jp.max(S)

@jax.jit
def jit_tightness(R):
    d = R.shape[0]
    return jp.linalg.norm((R @ R.T) - jp.eye(d))**2

def decode_norm(norm, jit=False):
    norm = norm if type(norm) != None else "p2"
    if type(norm) == str:
        if norm[0] == "p":
            p = int(norm[1:])
            return jax.jit(lambda A: jit_pnorm(A, p)) if jit else \
                   lambda A: pnorm(A, p)
    return norm

def frame_quantumness(R, S=None, norm=None):
    d, n = R.shape
    S = R if type(S) == type(None) else S
    P = np.abs(R.conj().T @ (np.tile(1/np.linalg.norm(S, axis=0), (d, 1))*S))**2
    return decode_norm(norm)(np.eye(n) - spectral_inverse(P))



def min_quantumness_real_frame_parallel(d, n=None, 
                                           norm="p2",\
                                           method="SLSQP",\
                                           tol=1e-26,\
                                           options={"ftol": 1e-26,\
                                                    "disp": False,\
                                                    "maxiter": 10000}):
    r = int(d*(d+1)/2)
    n = r if type(n) == type(None) else n
    norm = decode_norm(norm, jit=True)

    @jax.jit
    def wrapped_quantumness(V):
        R = V.reshape(d, n)
        P = jp.abs(R.T @ (jp.tile(1/jp.linalg.norm(R, axis=0), (d, 1))*R))**2
        return norm(jit_spectral_inverse(P, r))
    
    @jax.jit
    def wrapped_tightness(V):
        R = V.reshape(d,n)
        return jit_tightness(R)
    
    V = np.random.randn(d*n)
    result = sc.optimize.minimize(\
                wrapped_quantumness, V,\
                jac=jax.jit(jax.jacrev(wrapped_quantumness)),\
                constraints=[{"type": "eq", 
                              "fun": wrapped_tightness,
                              "jac": jax.jit(jax.jacrev(wrapped_tightness))}],\
                tol=tol,\
                options=options,\
                method=method)
    if np.isclose(result.fun, float("nan"), equal_nan=True):
        return min_quantumness_real_frame_parallel(d, n, p=p)
    return np.array(result.x.reshape(d, n))

def min_quantumness_real_parallel(d, n=None, 
                                     norm="p2",\
                                     method="SLSQP",\
                                     tol=1e-26,\
                                     options={"ftol": 1e-26,\
                                             "disp": False,\
                                             "maxiter": 10000}):
    r = int(d*(d+1)/2)
    n = r if type(n) == type(None) else n
    norm = decode_norm(norm, jit=True)

    @jax.jit
    def wrapped_quantumness(V):
        K = V.reshape(n, d, d)
        P = jp.einsum("aji, ajk, blk, bli -> ab", *[K]*4)/jp.tile(jp.einsum("aji, aji -> a", K, K), (n,1))
        return norm(jit_spectral_inverse(P, r))
    
    @jax.jit
    def wrapped_tightness(V):
        K = V.reshape(d*n, d)
        return jp.linalg.norm((K.T @ K) - jp.eye(d))**2

    V = np.random.randn(n*d**2)
    result = sc.optimize.minimize(\
                wrapped_quantumness, V,\
                jac=jax.jit(jax.jacrev(wrapped_quantumness)),\
                constraints=[{"type": "eq", 
                              "fun": wrapped_tightness,
                              "jac": jax.jit(jax.jacrev(wrapped_tightness))}],\
                tol=tol,\
                options=options,\
                method=method)
    if np.isclose(result.fun, float("nan"), equal_nan=True):
        return min_quantumness_real_parallel(d, n, p=p)
    K = result.x.reshape(n, d, d)
    return np.transpose(K, (0, 2, 1)) @ K

def min_quantumness(d, n=None, norm="p2", field="complex", parallel=False, rank1=True,\
                    method="SLSQP", tol=1e-26, options={"ftol": 1e-26,\
                                                        "disp": False,\
                                                        "maxiter": 10000}):
    if field == "real":
        if not parallel:
            if rank1:
                return min_quantumness_real_frame_parallel(d, n=n, norm=norm, method=method, tol=tol, options=options)
            else:
                return min_quantumness_real_parallel(d, n=n, norm=norm, method=method, tol=tol, options=options)
