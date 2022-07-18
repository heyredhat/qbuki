import numpy as np
from scipy.special import binom

from .grassmannian import *

def spin_basis(j, m):
    return np.eye(int(2*j+1))[int(j-m)]

def sigma_plus(j):
    return np.array([[np.sqrt(j*(j+1) - m*n) \
                          if m == n+1 else 0 \
                              for n in np.arange(j, -j-1, -1)]\
                                 for m in np.arange(j, -j-1, -1)])

def sigma_minus(j):
    return sigma_plus(j).conj().T

def sigma_z(j):
    return np.diag(np.array([m for m in np.arange(j, -j-1, -1)]))

def sigma_y(j):
    return (sigma_plus(j) - sigma_minus(j))/(2j)

def sigma_x(j):
    return (sigma_plus(j) + sigma_minus(j))/2

###

def majorana_polynomial(ket):
    j = (len(ket)-1)/2
    return np.polynomial.Polynomial(\
                    [(-1)**(j-m)*\
                     np.sqrt(binom(2*j, j-m))*\
                     spin_basis(j, m).conj() @ ket\
                         for m in np.arange(-j, j+1)])

def projective_roots(n, poly):
    r = poly.roots()
    return np.concatenate([r, np.repeat(np.inf, n-len(r))])

def majorana_roots(ket):
    return projective_roots(len(ket)-1, majorana_polynomial(ket))

def plane_to_sphere(z):
    if z == np.inf:
        return np.array([0,0,-1])
    else:
        return np.array([2*z.real, 2*z.imag, 1-z.real**2 - z.imag**2])/(1+z.real**2+z.imag**2)
    
def projective_stars(n, poly):
    return np.array([plane_to_sphere(r) for r in projective_roots(n, poly)])

def majorana_stars(ket):
    return projective_stars(len(ket)-1, majorana_polynomial(ket))

###

def spin_operator_to_plucker(O, k):
    n = len(O)
    P = plucker_basis(k, n)
    return P @ sum([reduce(np.kron, \
                               [O if i == j else np.eye(n) \
                                   for j in range(k)])\
                                       for i in range(k)]) @ P.T/k

def lower_until_zero(j, k, highest=None):
    N = int(binom(int(2*j+1), k))
    highest = highest if type(highest) == np.ndarray else basis(N, 0)
    rungs = [highest]
    L = spin_operator_to_plucker(sigma_minus(j), k)
    while not np.allclose(rungs[-1], np.zeros(N)):
        rungs[-1] = rungs[-1]/np.linalg.norm(rungs[-1])
        rungs.append(L @ rungs[-1])
    return np.array(rungs[:-1])

###

# From qutip
def _factorial_prod(N, arr):
    arr[:int(N)] += 1

def _factorial_div(N, arr):
    arr[:int(N)] -= 1

def _to_long(arr):
    prod = 1
    for i, v in enumerate(arr):
        prod *= (i+1)**int(v)
    return prod

def clebsch(j1, j2, j3, m1, m2, m3):
    if m3 != m1 + m2:
        return 0
    vmin = int(np.max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(np.min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    c_factor = np.zeros((int(j1 + j2 + j3 + 1)), np.int32)
    _factorial_prod(j3 + j1 - j2, c_factor)
    _factorial_prod(j3 - j1 + j2, c_factor)
    _factorial_prod(j1 + j2 - j3, c_factor)
    _factorial_prod(j3 + m3, c_factor)
    _factorial_prod(j3 - m3, c_factor)
    _factorial_div(j1 + j2 + j3 + 1, c_factor)
    _factorial_div(j1 - m1, c_factor)
    _factorial_div(j1 + m1, c_factor)
    _factorial_div(j2 - m2, c_factor)
    _factorial_div(j2 + m2, c_factor)
    C = np.sqrt((2.0 * j3 + 1.0)*_to_long(c_factor))

    s_factors = np.zeros(((vmax + 1 - vmin), (int(j1 + j2 + j3))), np.int32)
    sign = (-1) ** (vmin + j2 + m2)
    for i,v in enumerate(range(vmin, vmax + 1)):
        factor = s_factors[i,:]
        _factorial_prod(j2 + j3 + m1 - v, factor)
        _factorial_prod(j1 - m1 + v, factor)
        _factorial_div(j3 - j1 + j2 - v, factor)
        _factorial_div(j3 + m3 - v, factor)
        _factorial_div(v + j1 - j2 - m3, factor)
        _factorial_div(v, factor)
    common_denominator = -np.min(s_factors, axis=0)
    numerators = s_factors + common_denominator
    S = sum([(-1)**i * _to_long(vec) for i,vec in enumerate(numerators)]) * \
        sign / _to_long(common_denominator)
    return C * S

spherical_tensor_cache = {}

def spherical_tensor(j, l, m):
    if (j, l, m) in spherical_tensor_cache:
        return spherical_tensor_cache[(j, l, m)]
    T = np.array([[(-1.)**(j-m1-m)*clebsch(j, j, l, m1, -m2, m) \
                   for m1 in np.arange(j, -j-1, -1)]\
                          for m2 in np.arange(j, -j-1, -1)])
    spherical_tensor_cache[(j, l, m)] = T
    return T

def multipole_expansion(O):
    j = (len(O)-1)/2
    return np.array([[(spherical_tensor(j, l, m).conj().T @ O).trace() for m in np.arange(-l, l+1)] for l in np.arange(0, 2*j+1)])

###

def expect_xyz(ket):
    j = (len(ket)-1)/2
    return np.array([ket.conj() @ sigma_x(j) @ ket,\
                     ket.conj() @ sigma_y(j) @ ket,\
                     ket.conj() @ sigma_z(j) @ ket]).real

def rotate(j, theta, phi):
    return sc.linalg.expm((theta/2) * \
                          (np.exp(-1j*phi)*sigma_plus(j) -\
                           np.exp(1j*phi)*sigma_minus(j)))

def cartesian_to_spherical(xyz):
    x, y, z = xyz
    r = np.linalg.norm(xyz)
    return np.array([r,np.arccos(z/r),np.arctan2(y,x)])

###

def first_nonzero_in_multipole_expansion(ket):
    j = (len(ket)-1)/2
    MP = multipole_expansion(np.outer(ket, ket.conj()))
    for l in np.arange(0, 2*j+1):
        for m in np.arange(-l, l+1):
            c = MP[int(l)][int(m+l)]
            if not np.isclose(c,0) and m != 0:
                return [c, l, m]
            
def spectator_component(ket):
    if len(ket) == 1:
        return ket[0]
    j = (len(ket)-1)/2
    r, theta, phi = cartesian_to_spherical(expect_xyz(ket))
    ket_ = rotate(j, theta, phi) @ ket
    c, l, m = first_nonzero_in_multipole_expansion(ket_)
    ket__ = sc.linalg.expm(-1j*np.angle(c)*sigma_z(j)/m) @ ket_
    c2 = ket__[np.where(np.logical_not(np.isclose(ket__, 0)))[0][0]]
    return np.sqrt(ket.conj() @ ket)*np.exp(1j*np.angle(c2))

###

def check_1anticoherence(G):
    k, n = G.shape
    j = (n-1)/2
    return np.allclose(np.array([G @ sigma_x(j) @ G.conj().T,\
                                 G @ sigma_y(j) @ G.conj().T,\
                                 G @ sigma_z(j) @ G.conj().T]),0)
def qmatrix(f, t):
    k = len(f(0))
    return np.array([[f(0)[i].conj() @ f(t)[j] for j in range(k)] for i in range(k)])

def holonomy(f, t):
    U, D, V = np.linalg.svd(qmatrix(f, t))
    return U @ V

def rot_holonomy(G, R):
    k = G.shape[0]
    G_ = G @ R
    Q = np.array([[G[i].conj() @ G_[j] for j in range(k)] for i in range(k)])
    U, D, V = np.linalg.svd(Q)
    return U @ V

###

def sign(lst):
    parity = 1
    for i in range(0,len(lst)-1):
        if lst[i] != i:
            parity *= -1
            mn = min(range(i,len(lst)), key=lst.__getitem__)
            lst[i],lst[mn] = lst[mn],lst[i]
    return parity    

def principle_constellation_polynomial(G):
    k, n = G.shape
    P = [majorana_polynomial(g) for g in G]
    W = np.array([[P[i].deriv(m=j) for j in range(k)] for i in range(k)])
    return sum([sign(list(perm))*np.prod([W[i,p] for i, p in enumerate(perm)]) for perm in permutations(list(range(k)))]) 

def number_of_principal_stars(j, k):
    return 2*np.sum(np.arange(j-k+1, j+1))

###

def spin_coherent_state(j, z):
    if z == np.inf:
        return spin_coherent_state(j, 0)[::-1]
    return (1+z*z.conjugate())**(-j) * \
            sum([np.sqrt(binom(2*j, j-m))*z**(j-m)*spin_basis(j,m)\
                     for m in np.arange(-j, j+1)])

##

def integer(x):
    return np.equal(np.mod(x, 1), 0)

def spin_coherent_representation(ket):
    j = (len(ket)-1)/2
    m = majorana_polynomial(ket)
    def scs_rep(z):
        if z == np.inf:
            return ket[0]
        elif np.isclose(z, 0):
            return ket[-1]
        return ((z.conjugate()/z)/(1+z*z.conjugate()))**j * m(z)
    return scs_rep

###

def coherent_plane(j, k):
    d = int(2*j+1)
    return lambda z: np.eye(d)[:k,:]@sc.linalg.expm(-z*sigma_plus(j)) 

###

import jax
import jax.numpy as jp
from jax.config import config
config.update("jax_enable_x64", True)

def find_1anticoherent_subspace(j, k, max_iter=1000):
    d = int(2*j + 1)
    X, Y, Z = sigma_x(j), sigma_y(j), sigma_z(j)

    @jax.jit
    def one_anticoherence(V):
        R = (V[0:d*k] + 1j*V[d*k:]).reshape(k, d)
        return (jp.linalg.norm(R @ X @ R.conj().T) + \
                jp.linalg.norm(R @ Y @ R.conj().T) + \
                jp.linalg.norm(R @ Z @ R.conj().T)).real

    @jax.jit
    def orthogonality(V):
        R = (V[0:d*k] + 1j*V[d*k:]).reshape(k, d)
        return jp.linalg.norm((R @ R.conj().T) - jp.eye(k)).real
    
    for t in range(max_iter):
        try:
            V = np.random.randn(2*d*k)
            result = sc.optimize.minimize(one_anticoherence, V,\
                                          jac=jax.jit(jax.jacrev(one_anticoherence)),\
                                          tol=1e-23,\
                                          constraints=[{"type": "eq",\
                                                        "fun": orthogonality,\
                                                        "jac": jax.jit(jax.jacrev(orthogonality))}],\
                                          options={"disp": True,\
                                                   "maxiter": 5000},
                                          method="trust-constr")
            R = (result.x[0:d*k] + 1j*result.x[d*k:]).reshape(k, d)
            return R
        except:
            continue

##

