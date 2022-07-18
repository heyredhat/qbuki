import numpy as np
import scipy as sc
from scipy.stats import unitary_group
from itertools import combinations, permutations

from .utils import *

def random_grassmannian(k, n):
    return unitary_group.rvs(n)[:k]

def standard_grassmannian_form(G):
    return np.linalg.inv(G[:,:2]) @ G

def plucker_coordinate(I, G):
    return np.linalg.det(G[:, I])

def plucker_indices(k, n):
    return list(combinations(list(range(n)), k))

def plucker_coordinates(G):
    return np.array([plucker_coordinate(i, G) for i in plucker_indices(*G.shape)])

###

def __antisymmetrize__(a, b):
    return np.kron(a, b) - np.kron(b, a)

def antisymmetrize(*V):
    return reduce(__antisymmetrize__, V)

###

def plucker_basis(k, n):
    return np.array([antisymmetrize(*[basis(n, i) for i in I]) for I in plucker_indices(k, n)])

def plucker_inner_product(v, w):
    return np.linalg.det(v.conj() @ w.T)

def kplane_inner_product(v, w):
    return abs(plucker_inner_product(v,w))/\
                (np.sqrt(plucker_inner_product(v,v))*\
                 np.sqrt(plucker_inner_product(w,w)))