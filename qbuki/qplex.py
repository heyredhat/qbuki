import numpy as np
import scipy as sc
from scipy.special import hyp2f1
import multiprocessing as mp
from itertools import * 

from scipy.special import gamma
from scipy.stats import dirichlet
from scipy.spatial import ConvexHull
from scipy.spatial import HalfspaceIntersection

import matplotlib.pyplot as plt

import jax
import jax.numpy as jp
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

from .sics import *
from .weyl_heisenberg import *
from .random import *

####################################################################################################

def flatten(l):
    return [item for sublist in l for item in sublist]

####################################################################################################

def vol_simplex(d):
    return d/gamma(d**2)

def vol_psic(d):
    return np.sqrt((2*np.pi)**(d*(d-1))/(d**(d**2-2)*(d+1)**(d**2-1)))*\
           np.prod([gamma(i) for i in range(1, d+1)])/gamma(d**2)

####################################################################################################

def __make_surface_constraint__(d):
    @jax.jit
    def constraint(V):
        V = jp.abs(V)
        V = V/jp.sum(V)
        return (V@V - 2/(d*(d+1)))**2
    constraint_jac = jax.jit(jax.jacrev(constraint))
    return constraint, constraint_jac

def sample_qplex_surface_point(d, constraint=None, constraint_jac=None):
    if type(constraint) == type(None):
        constraint, constraint_jac = __make_surface_constraint__(d)
    V = np.random.uniform(size=d**2)
    result = sc.optimize.minimize(constraint, V,\
                              jac=constraint_jac,\
                              tol=1e-16,\
                              options={"disp": False,\
                                       "maxiter": 25})
    if not result.success:
        return sample_qplex_surface_point(d, constraint=constraint, constraint_jac=constraint_jac)
    p = np.abs(result.x)
    p = p/np.sum(p)
    return p

def sample_qplex(d, qplex_pts=None, n_qplex_pts=10000, from_surface=True):
    if type(qplex_pts) == type(None):
        qplex_pts = np.array([[1/d if i == j else  1/(d*(d+1))\
                                for j in range(d**2)]\
                                    for i in range(d**2)])
        if n_qplex_pts == d**2:
            return qplex_pts
    if from_surface:
        constraint, constraint_jac = __make_surface_constraint__(d)
    while len(qplex_pts) < n_qplex_pts:
        if from_surface:
            pt = sample_qplex_surface_point(d, constraint=constraint, constraint_jac=constraint_jac)
            inner_products = qplex_pts @ pt
            if np.all(inner_products >= 1/(d*(d+1))):
                qplex_pts = np.vstack([qplex_pts, pt])
        else:
            pt = np.random.dirichlet((1,)*d**2)
            inner_products = qplex_pts @ pt
            if np.all(inner_products >= 1/(d*(d+1)-1e-16)) \
               and np.all(inner_products <= 2/(d*(d+1))+1e-16) \
               and pt @ pt <= 2/(d*(d+1))+1e-16:
                qplex_pts = np.vstack([qplex_pts, pt])
    return qplex_pts

####################################################################################################

def sample_hilbert_qplex(d, n_qplex_pts=10000):
    fiducial = sic_fiducial(d)[:,0]
    R = np.array([displace(d, i, j) @ fiducial/np.sqrt(d) for j in range(d) for i in range(d)]).conj()
    qplex_pts = np.array([[1/d if i == j else  1/(d*(d+1)) for j in range(d**2)] for i in range(d**2)])
    if n_qplex_pts == d**2:
        return qplex_pts
    return np.vstack([qplex_pts, np.array([abs(R @ rand_ket(d)[:,0])**2 for i in range(n_qplex_pts-d**2)])])

####################################################################################################

def mc_batch(qplex_pts, batch):
    d = int(np.sqrt(qplex_pts.shape[1]))
    hits = 0
    np.random.seed()
    mc_pts = np.random.dirichlet((1,)*d**2, size=batch)
    whittled = mc_pts[np.linalg.norm(mc_pts, axis=1)**2 <= 2/(d*(d+1))]
    if len(whittled) != 0:
        C = np.apply_along_axis(np.all, 1, whittled @ qplex_pts.T >= 1/(d*(d+1)))
        D = dict(zip(*np.unique(C, return_counts=True)))
        hits += D[True] if True in D else 0
    return hits

def cvx_batch(qplex_pts, batch):
    d = int(np.sqrt(qplex_pts.shape[1]))
    hits = 0
    np.random.seed()
    mc_pts = np.random.dirichlet((1,)*d**2, size=batch)
    C = np.array([in_hull(qplex_pts, pt) for pt in mc_pts])
    D = dict(zip(*np.unique(C, return_counts=True)))
    hits += D[True] if True in D else 0
    return hits

def mc_qplex_vol(qplex_pts, n_mc_pts=50000, batch_size=5000, batch_func=None):
    batch_func = mc_batch if batch_func == None else batch_func
    d = int(np.sqrt(qplex_pts.shape[1]))
    n_batches = n_mc_pts // batch_size
    batches = [batch_size]*n_batches
    remaining = n_mc_pts - batch_size*n_batches
    if remaining != 0:
        batches.append(remaining)
    pool = mp.Pool(mp.cpu_count())
    future_res = [pool.apply_async(batch_func, (qplex_pts, batch)) for batch in batches]
    hits = sum([f.get() for f in future_res])
    pool.close()
    return vol_simplex(d)*hits/n_mc_pts

def mc_qplex_vol_error_bars(Q, n_mc_pts=100000, batch_size=5000, N=1000, batch_func=None):
    V = [mc_qplex_vol(Q, n_mc_pts=n_mc_pts, batch_size=batch_size, batch_func=batch_func) for n in range(N)]
    return (np.mean(V), np.std(V))

####################################################################################################

def cvx_qplex_vol(qplex_pts):
    n = qplex_pts.shape[1]
    I = np.eye(n) - np.ones(n)/n
    U, D, V = np.linalg.svd(I@I.T)
    k = np.count_nonzero(np.round(D, decimals=4))
    O = (np.sqrt(np.diag(D[:k])) @ V[:k]).T
    P = sum([np.outer(O[i], I[i]) for i in range(n)])
    return ConvexHull(np.einsum('...ij,...j', P, qplex_pts)).volume

####################################################################################################

def tconstraint(d, t):
    return hyp2f1(1, -t, d, -1)/(d*(d+1))**t

def tcompare(Q, t):
    return (tconstraint(int(np.sqrt(Q.shape[1])), t), np.sum((Q @ Q.T)**t, axis=0)/len(Q))

####################################################################################################

def consistent(Q):
    d = int(np.sqrt(Q.shape[1]))
    L = (1/(d*(d+1)))
    U = (2/(d*(d+1)))
    I = Q @ Q.T
    return np.all(I >= L-1e-16) and np.all(I <= U+1e-16) and \
           np.allclose(np.sum(Q, axis=1), np.ones(Q.shape[0])) and \
           np.all(Q >= 0) and np.all(Q <=1)

####################################################################################################

def qplex_halfspace_intersection(points):
    n = points.shape[0]
    d = int(np.sqrt(points.shape[1]))
    L = (1/(d*(d+1)))
    U = (2/(d*(d+1)))
    halfspaces = np.vstack([np.hstack([-points, L*np.ones(n).reshape(n,1)]),
                            np.hstack([points, -U*np.ones(n).reshape(n,1)])])
    return HalfspaceIntersection(halfspaces, interior_point(halfspaces))

####################################################################################################

def john_plex():
    d = 3
    n = d**2
    L = (1/(d*(d+1)))
    U = (2/(d*(d+1)))
    SIC = (1/d-L)*jp.eye(n) + L*jp.ones((n,n))
    MUBs = np.array(list(set(permutations([0]*3 + [1]*6))))/6
    return np.vstack([SIC, MUBs])

####################################################################################################

from scipy.optimize import linprog

def in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

####################################################################################################

def add_new_point_on_outsphere(qplex_pts, N=1):
    d = int(np.sqrt(qplex_pts.shape[1]))
    n = d**2
    L = (1/(d*(d+1)))-1e-16
    U = (2/(d*(d+1)))+1e-16

    @jax.jit
    def novelty(V):
        P = V.reshape(N, n)
        P = (P.T/jp.sum(P,axis=1)).T
        return -jp.sum(jp.linalg.norm(qplex_pts - P, axis=1))

    @jax.jit
    def sphere(V):
        P = V.reshape(N, n)
        P = (P.T/jp.sum(P,axis=1)).T
        return (jp.linalg.norm(jp.diag(P @ P.T) - U*jp.ones(N))) #/ 10000000

    @jax.jit
    def consistent_min(V):
        P = V.reshape(N, n)
        P = (P.T/jp.sum(P,axis=1)).T
        Q = jp.vstack([qplex_pts, P])
        return (Q @ Q.T).flatten() - L

    @jax.jit
    def consistent_max(V):
        P = V.reshape(N, n)
        P = (P.T/jp.sum(P,axis=1)).T
        Q = jp.vstack([qplex_pts, P])
        return -(Q @ Q.T).flatten() + U

    result = sc.optimize.minimize(sphere, np.random.randn(N*n),\
                        jac=jax.jit(jax.jacrev(sphere)),\
                        bounds=[(0,1)]*(N*n),
                        tol=1e-16,\
                        constraints=[{"type": "eq",\
                                    "fun": novelty,\
                                    "jac": jax.jit(jax.jacrev(novelty))},
                                    {"type": "ineq",\
                                    "fun": consistent_min,\
                                    "jac": jax.jit(jax.jacrev(consistent_min))},\
                                    {"type": "ineq",\
                                    "fun": consistent_max,\
                                    "jac": jax.jit(jax.jacrev(consistent_max))}],\
                        options={"maxiter": 1000, "disp": True})
    P = jp.abs(result.x.reshape(N, n))
    P = (P.T/np.sum(P,axis=1)).T
    return P

def add_if_consistent(qplex_pts, pts):
    d = int(np.sqrt(qplex_pts.shape[1]))
    L = (1/(d*(d+1)))-1e-16
    U = (2/(d*(d+1)))+1e-16
    J = qplex_pts[:]
    for p in pts:
        I = qplex_pts @ p
        if np.all(I > L) and np.all(I < U):
            J = np.vstack([J, p])
    return J

def permutations_of(v):
    P = np.array(list(set(permutations(flatten([[i]*v[i][1] for i in range(len(v))])))), dtype=np.float128)
    for i, _ in enumerate(v):
        np.place(P, P==i, _[0])
    return P

def add_if_consistent(Q, pts):
    d = int(np.sqrt(Q.shape[1]))
    L = (1/(d*(d+1)))
    U = (2/(d*(d+1)))
    epsilon = 1e-16
    for p in pts:
        I = Q @ p
        if np.all(I >= L-epsilon) and np.all(I <= U+epsilon):
            Q = np.vstack([Q, p])
    return Q

def add_if_consistent_recursive(V):
    def __add_if_helper__(A, B):
        if len(B) == 1:
            return add_if_consistent(A, B[0])
        return __add_if_helper__(__add_if_helper__(A, B[:1]), B[1:])
    return __add_if_helper__(V[0], V[1:])

def non_increasing(a):
    return np.all(np.diff(a) <= 0)