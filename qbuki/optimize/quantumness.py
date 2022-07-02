import scipy as sc
import numpy as np
from functools import partial

import jax
import jax.numpy as jp
from jax.config import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

from ..utils import *
from ..povm_utils import *

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

def decode_norm(norm):
    norm = norm if type(norm) != None else "p2"
    if type(norm) == str:
        if norm[0] == "p":
            p = int(norm[1:])
            return jax.jit(lambda A: jit_pnorm(A, p))
    return norm

def min_quantumness(d, n=None, field="complex",
                               norm="p2",\
                               rank1=True,\
                               parallel=True,\
                               method="SLSQP",\
                               tol=1e-26,\
                               options={"disp": False,\
                                        "maxiter": 10000},\
                               max_iter=100,\
                               return_params=False):
    r = int(d*(d+1)/2) if field == "real" else int(d**2)
    n = r if type(n) == type(None) else n
    norm_func = decode_norm(norm)

    if rank1:
        if field == "complex":
            decode_params = jax.jit(lambda V: [(V[:d*n] + 1j*V[d*n:]).reshape(d, n)]*2) if parallel else\
                            jax.jit(lambda V: [(V[:d*n] + 1j*V[d*n:2*d*n]).reshape(d, n),\
                                               (V[2*d*n:3*d*n] + 1j*V[3*d*n:]).reshape(d, n)])
            initial_params = (lambda: np.random.randn(2*d*n)) if parallel else (lambda: np.random.randn(4*d*n)) 
        elif field == "real":
            decode_params = jax.jit(lambda V: [V.reshape(d, n)]*2) if parallel else\
                            jax.jit(lambda V: [V[:d*n].reshape(d, n), V[d*n:].reshape(d, n)])
            initial_params = (lambda: np.random.randn(d*n)) if parallel else (lambda: np.random.randn(2*d*n))

        def final_decode(V):
            if parallel:
                return frame_povm(np.array(decode_params(V)[0]))
            else:
                R, S = decode_params(V)
                S = jp.tile(1/jp.linalg.norm(S, axis=0), (d, 1))*S
                return frame_povm(np.array(R)), frame_povm(np.array(S))

        @jax.jit
        def wrapped_quantumness(V):
            R, S = decode_params(V)
            S = jp.tile(1/jp.linalg.norm(S, axis=0), (d, 1))*S
            P = jp.abs(R.conj().T @ S)**2
            return norm_func(jit_spectral_inverse(P, r))

        @jax.jit
        def wrapped_tightness(V):
            R, S = decode_params(V)
            return jp.linalg.norm((R @ R.conj().T) - jp.eye(d))**2
    else:
        if field == "complex":
            decode_params = jax.jit(lambda V: [(V[:n*d**2] + 1j*V[n*d**2:]).reshape(n, d, d)]*2) if parallel else\
                            jax.jit(lambda V: [(V[:n*d**2] + 1j*V[n*d**2:2*n*d**2]).reshape(n, d, d),\
                                               (V[2*n*d**2:3*n*d**2] + 1j*V[3*n*d**2:]).reshape(n, d, d)])
            initial_params = (lambda: np.random.randn(2*n*d**2)) if parallel else (lambda: np.random.randn(4*n*d**2)) 
        elif field == "real":
            decode_params = jax.jit(lambda V: [V.reshape(n, d, d)]*2) if parallel else\
                            jax.jit(lambda V: [V[:n*d**2].reshape(n, d, d), V[n*d**2:].reshape(n, d, d)]) 
            initial_params = (lambda: np.random.randn(n*d**2)) if parallel else (lambda: np.random.randn(2*n*d**2)) 

        def final_decode(V):
            make = lambda K: np.array([k.conj().T @ k for k in K])
            if parallel:
                return make(decode_params(V)[0])
            else:
                R, S = [make(x) for x in decode_params(V)]
                return R, np.array([s/s.trace() for s in S])

        @jax.jit
        def wrapped_quantumness(V):
            KR, KS = decode_params(V)
            P = jp.einsum("aji, ajk, blk, bli -> ab", KR.conj(), KR, KS.conj(), KS)/jp.tile(jp.einsum("aji, aji -> a", KS.conj(), KS), (n,1))
            return norm_func(jit_spectral_inverse(P, r))
        
        @jax.jit
        def wrapped_tightness(V):
            KR, KS = decode_params(V)
            KR = KR.reshape(d*n, d)
            return jp.linalg.norm((KR.conj().T @ KR) - jp.eye(d))**2

    for t in range(max_iter):
        result = sc.optimize.minimize(\
                    wrapped_quantumness, initial_params(),\
                    jac=jax.jit(jax.jacrev(wrapped_quantumness)),\
                    constraints=[{"type": "eq", 
                                "fun": wrapped_tightness,
                                "jac": jax.jit(jax.jacrev(wrapped_tightness))}],\
                    tol=tol,\
                    options=options,\
                    method=method)
        if not np.isclose(result.fun, float("nan"), equal_nan=True):
            if return_params:
                X = [np.array(x) for x in decode_params(result.x)]
                return X[0] if parallel else X
            return final_decode(result.x)
                