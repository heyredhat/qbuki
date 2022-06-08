import numpy as np
import scipy as sc

from .utils import *

class Operators:
    def __init__(self, E):
        E = E.E if type(E) == Operators else np.array(E)
        if len(E.shape) == 2:
            E = E.reshape(1, *E.shape)
        self.E = E
        self.inverted = False

    def __len__(self):
        r"""
        Number of operators.
        """
        return self.E.shape[0]

    def dim(self):
        r"""
        Dimensionality of the operators.
        """
        return self.E.shape[-1]
        
    def __getitem__(self, key):
        r"""
        Get operator by index.
        """
        return self.E[key]

    def __setitem__(self, key, value):
        r"""
        Set operator by index.
        """
        self.E[key] = value
    
    def __iter__(self):
        r"""
        Iterator over operators.
        """
        return self.E.__iter__()

    def bias(self):
        r"""
        Returns the traces of each operator.
        """
        return np.trace(self.E, axis1=1, axis2=2)
    
    def conj(self):
        r"""
        Conjugate each operator.
        """
        return Operators(self.E.conj())
    
    @property
    def T(self):
        r"""
        Transpose each operator.
        """
        return Operators(np.transpose(self.E, (0,2,1)))

    def __matmul__(self, other):
        r"""
        Pairwise matrix multiplication of two set of of operators.
        """
        return Operators(self.E @ other.E)
    
    def __add__(self, other):
        r"""
        Concatenates to sets of operators.
        """
        return Operators(np.concatenate([self.E, other.E]))

    def __and__(self, other):
        r"""
        Tensor product of elements in self and elements in other.
        """
        return Operators(np.kron(self[:], other[:]))

    def __mul__(self, other):
        r"""
        Scalar multiplication on each operator.
        """
        return Operators(other*self[:])
    
    def __rmul__(self, other):
        r"""
        Scalar multiplication on each operator.
        """
        return self.__mul__(other)
    
    def __truediv__(self, other):
        r"""
        Scalar division on each operator.
        """
        return self.__mul__(1/other)
    
    def upgrade(self, i, dims):
        r"""
        Upgrades operators to act on the i'th place of a tensor product with structure dims.
        """
        return Operators([upgrade(e, i, dims) for e in self[:]])

    def flatten(self):
        r"""
        Returns the matrix whose columns are vectorized operators.
        """
        return self.E.reshape(self.E.shape[0], np.prod(self.E.shape[1:])).T

    def __invert__(self):
        r"""
        Toggles inversion: ~A.
        """
        self.inverted = True if not self.inverted else False
        return self

    def __inverted__(self, A):
        r"""
        Returns A if not inverted else returns the spectral inverse of A, and then uninverts.
        """
        A = A if not self.inverted else spectral_inverse(A)
        self.inverted = False if self.inverted else self.inverted
        return A

    def __xor__(self, other):
        r"""
        A^B gives the (inverted) matrix of traces between elements of self and elements of other.
        """
        return self.__inverted__(self.flatten().conj().T @ other.flatten())
    
    def __or__(self, other):
        r"""
        A|B gives the (inverted) matrix of traces between elements of self and elements of other, the latter normalized to trace 1.
        """
        return self.__inverted__(self.flatten().conj().T @ (other.flatten()/other.bias()))

    def __lshift__(self, other):
        r"""
        A << rho expands rho in terms of A. 
        A << p reconstructs rho from p.
        """
        if len(other.shape) == 1 or other.shape[1] == 1:
            return (self.flatten() @ (~self^self) @ other).reshape(self.E.shape[1:])
        else:
            return (self.flatten().conj().T @ other.flatten()).reshape(len(self), 1)
    
    def superoperator(self):
        r"""
        Returns the superoperator of the elements of self interpreted as a set of Kraus operators.
        """
        return self.__inverted__(sum(kron(e, e.conj()) for e in self[:]))

    def __lt__(self, other):
        r"""
        A < rho applies A to rho, where A is interpreted as a set of Kraus operators forming a channel.
        Similarly A < B applies A to each element of B.
        """
        return (self.superoperator() @ other.flatten()).reshape(other.shape) if type(other) == np.ndarray\
                else Operators((self.superoperator() @ other.flatten()).T.reshape(other.E.shape))
    
    def __floordiv__(self, other):
        r"""
        Returns the (inverse) frame superoperator where other is the post measurement states.
        """
        return self.__inverted__(self.flatten() @ other.flatten().conj().T)

    def __pow__(self, other):
        r"""
        Returns dual elements with respect to other.
        E.g. if self is POVM and other is post-measurement states, this gives dual POVM elements.
        If self is post-measurement states and other is POVM, this gives dual post measurement elements.
        Together they give the dual reference measurement.
        """
        return Operators(((~self // other) @ self.flatten()).T.reshape(self.E.shape))

    def __mod__(self, f):
        r"""
        Applies function f to each operator.
        """
        return Operators(np.array([f(e) for e in self[:]]))

def quantumness(A, B, p=2):
    S = np.linalg.svd(np.eye(len(A)) - (~A|B), compute_uv=False)
    return np.sum(S**p)**(1/p) if p != np.inf else np.max(S)
