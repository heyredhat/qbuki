import numpy as np
import scipy as sc
from functools import reduce

def normalize(v):
    r"""
    Normalizes vector.
    """
    return v/np.linalg.norm(v)

def basis(d, i):
    r"""
    Returns basis vectors.
    """
    return np.eye(d)[i]

def kron(*args):
    r"""
    Numpy's kronecker product with a variable number of arguments.
    """
    return reduce(lambda x, y: np.kron(x, y), args)

def upgrade(O, i, dims):
    return kron(*[O if i == j else np.eye(d) for j, d in enumerate(dims)])

def pnorm(A, p=2):
    S = np.linalg.svd(A, compute_uv=False)
    return np.sum(S**p)**(1/p) if p != np.inf else np.max(S)

def paulis():
    r"""
    Returns Pauli matrices.
    """
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    return np.array([np.eye(2), X, Y, Z])

def gelmann_basis(d):
    r"""
    Hermitian operator basis.
    """
    def h_helper(d,k):
        if k == 1:
            return np.eye(d)
        if k > 1 and k < d:
            return sc.linalg.block_diag(h_helper(d-1, k), [0])
        if k == d:
            return np.sqrt(2/(d*(d+1)))*sc.linalg.block_diag(np.eye(d-1), [1-d])

    E = [[np.zeros((d,d)) for k in range(d)] for j in range(d)]
    for j in range(d):
        for k in range(d):
            E[j][k][j,k] = 1
    F = []
    for j in range(d):
        for k in range(d):
            if k < j:
                F.append(E[k][j] + E[j][k])
            elif k > j:
                F.append(-1j*(E[j][k] - E[k][j]))
    F.extend([h_helper(d, k) for k in range(1,d+1)])
    return np.array([f/np.sqrt((f@f).trace()) for f in F])

def fft_matrix(d):
    r"""
    Finite fourier transform matrix.
    """
    w = np.exp(2*np.pi*1j/d)
    return np.array([[w**(i*j) for j in range(d)] for i in range(d)])/np.sqrt(d)

def full_rank_decomposition(M):
    U, D, V = np.linalg.svd(M)
    m = (np.isclose(D, 0)).argmax()
    m = m if m != 0 else len(D)
    sqrtD = np.sqrt(np.diag(D[:m]))
    return U[:,:m] @ sqrtD, sqrtD @ V[:m,:]

def spectral_inverse(M):
    r"""
    Spectral/group inverse.
    """
    B, C = full_rank_decomposition(M)
    return (B @ np.linalg.matrix_power(C @ B, -2) @ C)

def partial_trace_kraus(keep, dims):
    r"""
    Constructs the Kraus map corresponding to the partial trace. Takes `keep` which is a single index or list of indices denoting
    subsystems to keep, and a list `dims` of dimensions of the overall tensor product Hilbert space. 

    For illustration, to trace over the $i^{th}$ subsystem of $n$, one would construct Kraus operators:

    $$ \hat{K}_{i} = I^{\otimes i - 1} \otimes \langle i \mid \otimes I^{\otimes n - i}$$.
    """
    if type(keep) == int:
        keep = [keep]
    trace_over = [i for i in range(len(dims)) if i not in keep]
    indices = [{trace_over[0]:t} for t in range(dims[trace_over[0]])]
    for i in trace_over[1:]:
        new_indices = []
        for t in range(dims[i]):
            new_indices.extend([{**j, **{i: t}} for j in indices])
        indices = new_indices
    return np.array([kron(*[np.eye(d) if i in keep else basis(d, index[i]) for i, d in enumerate(dims)]) for index in indices])

def state_space_dimension(d, field):
    r"""
    Returns the dimension of the state space for a give pure state space dimension and number field.
    """
    if field == "complex":
        n = int(d**2)
    elif field == "real":
        n = int(d*(d+1)/2)
    return n

def stereographic_projection(X, pole=None):
    if type(pole) == type(None):
        pole = np.eye(len(X))[-1]
    if np.isclose(X@pole, 1):
        return np.inf
    return (pole + (X - pole)/(1-X@pole))[:-1]

def reverse_stereographic_projection(X, pole=None):
    if type(X) != np.ndarray and X == np.inf:
        return pole
    if type(pole) == type(None):
        pole = np.eye(len(X)+1)[0]
    X = np.append(X, 0)
    return ((np.linalg.norm(X)**2 - 1)/(np.linalg.norm(X)**2 +1))*pole + (2/(np.linalg.norm(X)**2 +1))*X
