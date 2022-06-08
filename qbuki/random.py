import numpy as np
import scipy as sc
import scipy.linalg
from scipy.stats import ortho_group
from scipy.stats import unitary_group

import jax
import jax.numpy as jp
from jax.config import config
config.update("jax_enable_x64", True)

from .utils import *
from .povm_utils import *

def rand_ginibre(*shape, field="complex"):
    r"""
    Random sample from Ginibre ensemble.
    """
    if field == "complex":
        return np.random.randn(*shape) + 1j*np.random.randn(*shape)
    elif field == "real":
        return np.random.randn(*shape)

def rand_ket(d, field="complex"):
    r"""
    Random ket.
    """
    return normalize(rand_ginibre(d, 1, field=field))

def rand_ket_hs(d, field="complex"):
    if field == "complex": 
        return (unitary_group.rvs(d) @ np.eye(d)[0]).reshape(d, 1)
    elif field == "real":
        return ortho_group.rvs(d) @ np.eye(d)[0].reshape(d, 1)
        
def rand_herm(d, field="complex"):
    r"""
    Random Hermitian (symmetric) matrix.
    """
    X = rand_ginibre(d, d, field=field)
    return (X + X.conj().T)/2

def rand_dm(d, r=None, field="complex"):
    r"""
    Random density matrix.
    """
    r = r if type(r) != type(None) else d
    X = rand_ginibre(d, r, field=field)
    rho = X @ X.conj().T
    return rho/rho.trace()

def rand_unitary(d, field="complex"):
    """
    Random unitary (orthogonal) matrix.
    """
    if field == "complex":
        return unitary_group.rvs(d)
    elif field == "real":
        return ortho_group.rvs(d)

def rand_povm(d, n=None, r=None, field="complex"):
    r"""
    Generates a Haar distributed random POVM for a Hilbert space of dimension $d$, 
    with $n$ elements, and with rank $m$.

    $m$ must satisfy $d \leq mn$, and defaults to $m=1$, giving rank-1 POVM elements.

    $n$ defaults to $d^2$ if complex, $\frac{d(d+1)}{2}$ if real.
    """
    n = n if type(n) != type(None) else state_space_dimension(d, field)
    r = r if type(r) != type(None) else d
    if field == "complex":
        povm = np.zeros((n, d, d), dtype=np.complex128) 
        S = np.zeros(d, dtype=np.complex128) 
    elif field == "real":
        povm = np.zeros((n, d, d))
        S = np.zeros(d)

    for i in range(n):
        Xi = rand_ginibre(d, r, field=field)
        Wi = Xi @ Xi.conjugate().T
        povm[i, :, :] = Wi
        S = S + Wi
    S = sc.linalg.fractional_matrix_power(S, -1/2)
    for i in range(n):
        Wi = np.squeeze(povm[i, :, :])
        povm[i, :, :] = S @ Wi @ S
    return povm

def rand_effect(d, n=None, r=None, field="complex"):
    r"""
    Generates a Haar distributed random POVM effect of Hilbert space dimension $d$, 
    as if it were part of a POVM of $n$ elements with rank $m$. 
    """
    n = n if type(n) != type(None) else state_space_dimension(d, field)
    r = r if type(r) != type(None) else d

    X = rand_ginibre(d, r, field=field)
    W = X @ X.conjugate().T
    Y = rand_ginibre(d, (n-1)*r, field=field)
    S = W + Y @ Y.conjugate().T
    S = sc.linalg.fractional_matrix_power(S, -1/2)
    return S @ W @ S.conjugate().T

def rand_ftf(d, n=None, field="complex"):
    """
    Random tight frame.
    """
    n = n if type(n) != type(None) else state_space_dimension(d, field)
    if field == "complex":
        R = np.random.randn(d, n) + 1j*np.random.randn(d, n) 
    elif field == "real":
        R = np.random.randn(d, n)
    return tighten_frame(R)

def rand_funtf(d, n=None, field="complex", rtol=1e-15, atol=1e-15):
    r"""
    Random finite unit norm tight frame.
    """
    n = n if type(n) != type(None) else state_space_dimension(d, field)
    if field == "complex":
        R = np.random.randn(d, n) + 1j*np.random.randn(d, n) 
    elif field == "real":
        R = np.random.randn(d, n)
    done = False
    while not (np.allclose(R @ R.conj().T, (n/d)*np.eye(d), rtol=rtol, atol=atol) and\
               np.allclose(np.linalg.norm(R, axis=0), np.ones(n), rtol=rtol, atol=atol)):
        R = sc.linalg.polar(R)[0]
        R = np.array([state/np.linalg.norm(state) for state in R.T]).T
    return sc.linalg.polar(R)[0]

def rand_kraus(d, n, field="complex"):
    r"""
    Random Kraus operators.
    """
    G = [rand_ginibre(d, d, field=field) for i in range(n)]
    H = sum([g.conj().T @ g for g in G])
    S = sc.linalg.fractional_matrix_power(H, -1/2)
    return np.array([g @ S for g in G])

def rand_probs(n, m):
    r"""m random probability vectors of length n."""
    return np.random.dirichlet((1,)*n, size=m).T

def rand_probs_table(m, n, r):
    if r < 2:
        raise Exception("r must be > 1")
    P = np.vstack([np.ones((1,n)), np.random.uniform(low=0, high=1, size=(m-1, n))])
    r = r - 1

    @jax.jit
    def obj(V):
        A = V[:(m-1)*r].reshape(m-1, r)
        B = V[(m-1)*r:].reshape(r, n)
        AB = jp.vstack([np.ones((1,n)), A@B])
        return jp.linalg.norm(AB - P)

    @jax.jit
    def consistency_max(V):
        A = V[:(m-1)*r].reshape(m-1, r)
        B = V[(m-1)*r:].reshape(r, n)
        AB = jp.vstack([np.ones((1,n)), A@B])
        return -(AB).flatten() + 1

    V = np.random.randn((m-1)*r + r*n)
    result = sc.optimize.minimize(obj, V,\
                                  jac=jax.jit(jax.jacrev(obj)),\
                                  tol=1e-16,\
                                  constraints=[{"type": "ineq",\
                                                "fun": consistency_max,\
                                                "jac": jax.jit(jax.jacrev(consistency_max))}],\
                                  options={"maxiter": 5000},
                                  method="SLSQP")
    A = result.x[:(m-1)*r].reshape(m-1, r)
    B = result.x[(m-1)*r:].reshape(r, n)
    AB = np.vstack([np.ones((1,n)), A@B])
    if not (np.all(AB >= 0) and np.all(AB <= 1)):
        return rand_probs_table(m, n, r+1)
    else:
        return AB

def rand_quantum_probs_table(d, m, n, r=1, field="complex"):
    effects = [np.eye(d)] + [rand_effect(d, r=r, field=field) for _ in range(m-1)]
    states = [rand_dm(d, r=r, field=field) for _ in range(n)]
    return np.array([[(e @ s).trace() for s in states] for e in effects]).real

def sample_convex_hull(hull, n):
    # https://stackoverflow.com/questions/59073952/how-to-get-uniformly-distributed-points-in-convex-hull
    points = hull.points
    dims = points.shape[-1]
    hull = points[hull.vertices]
    deln = hull[sc.spatial.Delaunay(hull).simplices]

    vols = np.abs(np.linalg.det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)    
    sample = np.random.choice(len(vols), size = n, p = vols / vols.sum())

    return np.einsum('ijk, ij -> ik', deln[sample], sc.stats.dirichlet.rvs([1]*(dims + 1), size = n))