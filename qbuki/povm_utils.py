import numpy as np
import scipy as sc

from .utils import *

def vn_povm(H):
    r"""
    Returns a P(O)VM corresponding to the eigenstates of a Hermitian operator.
    """
    return np.array([np.outer(v, v.conj().T) for v in np.linalg.eig(H)[1][::-1]])

def tighten(R):
    r"""
    Tightens frame.
    """
    return sc.linalg.polar(R)[0]

def frame_povm(R):
    r"""
    Lifts tight frame to POVM.
    """
    return np.array([np.outer(r, r.conj()) for r in R.T])

def squish(E):
    S = sc.linalg.fractional_matrix_power(sum(E), -1/2)
    return np.array([S @ e @ S for e in E])

def complete(E):
    d = E.shape[-1]
    return np.vstack([E, (np.eye(d) - sum(E)).reshape(1,d,d)])

def dilate(E):
    EE = []
    mapping = {}
    for i, e in enumerate(E):
        L, V = np.linalg.eig(e)
        mapping[i] = []
        for j in range(len(L)):
            if not np.isclose(L[j], 0):
                EE.append(L[j]*np.outer(V[j], V[j].conj()))
                mapping[i].append(len(EE)-1)
    return np.array(EE)

def coarse_grain(E, mapping):
    return np.array([sum([E[v] for v in V]) for k, V in mapping.items()])

def sample_from_povm(E, rho, n=1):
    p = np.array([(e@rho).trace() for e in E]).real
    return np.random.choice(list(range(len(E))), size=n, p=p)

def implement_povm(E):
    n, d = len(E), E[0].shape[0]
    V = sum([np.kron(basis(n, i), sc.linalg.sqrtm(E[i])) for i in range(n)]).T
    Q, R = np.linalg.qr(V, mode="complete")
    return -Q.conj()

def discriminator_povm(a, b):
    r"""
    Returns a non informationally complete POVM which has the special property
    of distinguishing between two arbitrary states $\mid a \rangle$ and $\mid b\rangle$, which are not necessarily orthogonal (which is impossible with a standard PVM).

    It has three elements:

    $$ \hat{F}_{a} = \frac{1}{1+\mid\langle a \mid b \rangle\mid}(\hat{I} - \mid b \rangle \langle b \mid) $$
    $$ \hat{F}_{b} = \frac{1}{1+\mid\langle a \mid b \rangle\mid}(\hat{I} - \mid a \rangle \langle a \mid) $$
    $$ \hat{F}_{?} = \hat{I} - \hat{F}_{a} - \hat{F}_{b} $$

    The first tests for "not B", the second tests for "not A", and the third outcome represents an inconclusive result.
    """
    d = a.shape[0]
    p = abs(a.conj().T @ b)
    Fa = (1/(1+p))*(np.eye(d) - b @ b.conj().T)
    Fb = (1/(1+p))*(np.eye(d) - a @ a.conj().T)
    Fq = np.eye(d) - Fa - Fb
    return np.array([Fa, Fb, Fq])
