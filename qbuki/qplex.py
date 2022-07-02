import numpy as np
import scipy as sc
import multiprocessing as mp

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

def sample_qplex_from_surface_points(d, qplex_pts=None, n_qplex_pts=10000):
    if type(qplex_pts) == type(None):
        qplex_pts = np.array([[1/d if i == j else  1/(d*(d+1))\
                                for j in range(d**2)]\
                                    for i in range(d**2)])
        if n_qplex_pts == d**2:
            return qplex_pts
    constraint, constraint_jac = __make_surface_constraint__(d)
    while len(qplex_pts) < n_qplex_pts:
        pt = sample_qplex_surface_point(d, constraint=constraint, constraint_jac=constraint_jac)
        inner_products = qplex_pts @ pt
        if np.all(inner_products >= 1/(d*(d+1))):
            qplex_pts = np.vstack([qplex_pts, pt])
    return qplex_pts

def qplex_moment_error(d, qplex_pts, moment_cutoff=10):
    n = qplex_pts.shape[0]
    empirical = np.array([np.sum(qplex_pts**m, axis=0)/n for m in range(1, moment_cutoff+1)])
    analytic = np.array([np.ones(d**2)/(d**m * sc.special.binom(m+d-1, d-1)) for m in range(1, moment_cutoff+1)])
    return np.linalg.norm(empirical - analytic)**2

def sample_centered_qplex_from_surface_points(d, qplex_pts=None, n_qplex_pts=10000, n_potentials=50):
    if type(qplex_pts) == type(None):
        qplex_pts = np.array([[1/d if i == j else  1/(d*(d+1))\
                                for j in range(d**2)]\
                                    for i in range(d**2)])
        if n_qplex_pts == d**2:
            return qplex_pts     
    constraint, constraint_jac = __make_surface_constraint__(d)
    C = np.ones(d**2)/d**2                       
    while len(qplex_pts) < n_qplex_pts:
        potentials = []
        while len(potentials) < n_potentials:
            pt = sample_qplex_surface_point(d, constraint=constraint, constraint_jac=constraint_jac)
            inner_products = qplex_pts @ pt
            if np.all(inner_products >= 1/(d*(d+1))):
                potentials.append(pt)
        potential_centeredness = [qplex_moment_error(d, np.vstack([qplex_pts, potential])) for potential in potentials]
        qplex_pts = np.vstack([qplex_pts, potentials[np.argmin(potential_centeredness)]])
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
    mc_pts = np.random.dirichlet((1,)*d**2, size=batch)
    whittled = mc_pts[np.linalg.norm(mc_pts, axis=1)**2 <= 2/(d*(d+1))]
    if len(whittled) != 0:
        C = np.apply_along_axis(np.all, 1, whittled @ qplex_pts.T >= 1/(d*(d+1)))
        D = dict(zip(*np.unique(C, return_counts=True)))
        hits += D[True] if True in D else 0
    return hits

def mc_qplex_vol(qplex_pts, n_mc_pts=50000, batch_size=5000):
    d = int(np.sqrt(qplex_pts.shape[1]))
    n_batches = n_mc_pts // batch_size
    batches = [batch_size]*n_batches
    remaining = n_mc_pts - batch_size*n_batches
    if remaining != 0:
        batches.append(remaining)

    pool = mp.Pool(mp.cpu_count())
    future_res = [pool.apply_async(mc_batch, (qplex_pts, batch)) for batch in batches]
    hits = sum([f.get() for f in future_res])
    return vol_simplex(d)*hits/n_mc_pts

####################################################################################################

def cvx_qplex_vol(qplex_pts):
    n = qplex_pts.shape[1]
    I = np.eye(n) - np.ones(n)/n
    U, D, V = np.linalg.svd(I@I.T)
    k = np.count_nonzero(np.round(D, decimals=4))
    O = (np.sqrt(np.diag(D[:k])) @ V[:k]).T
    P = sum([np.outer(O[i], I[i]) for i in range(n)])
    return ConvexHull(np.einsum('...ij,...j', P, qplex_pts)).volume
