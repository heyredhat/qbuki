import numpy as np
import scipy as sc
from scipy.spatial import ConvexHull
from scipy.spatial import HalfspaceIntersection
from functools import partial

import jax
import jax.numpy as jp
from jax.config import config
config.update("jax_enable_x64", True)

import cvxpy as cp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors

from .utils import *
from .random import *

def rank_approximation(P, k, sd=1, tol=1e-6, max_iter=1000, verbose=False):
    m, n = P.shape
    S = np.random.randn(m, k)
    E = np.random.randn(k, n)
    W = np.eye(m*n)/sd**2 
    t = 0; last = None
    while t < max_iter:
        try:
            U = np.kron(E, np.eye(m)) @ W @ np.kron(E.T, np.eye(m))
            V = -2*np.kron(E, np.eye(m)) @ W @ P.T.flatten()
            Y = np.kron(E.T,np.eye(m))

            Svec = cp.Variable(k*m)
            Sprob = cp.Problem(cp.Minimize(cp.quad_form(Svec, U) + V.T @ Svec), [0 <= Y @ Svec, Y @ Svec <= 1])
            Sresult = Sprob.solve()
            S = Svec.value.reshape(k, m).T

            U = np.kron(np.eye(n), S).T @ W @ np.kron(np.eye(n), S)
            V = -2*np.kron(np.eye(n), S).T @ W @ P.T.flatten()
            Y = np.kron(np.eye(n), S)

            Evec = cp.Variable(k*n)
            Eprob = cp.Problem(cp.Minimize(cp.quad_form(Evec, U) + V.T @ Evec), [0 <= Y @ Evec, Y @ Evec <= 1])
            Eresult = Eprob.solve()
            E = Evec.value.reshape(n,k).T
            if verbose and t % 100 == 0:
                print("%d: chi_S = %f | chi_E = %f " % (t, Sresult, Eresult))

            if type(last) != type(None) and abs(Sresult-last[0]) <= tol and abs(Eresult-last[1]) <= tol:
                break
            last = [Sresult, Eresult]
        except:
            S = np.random.randn(m, k)
            E = np.random.randn(k, n)
            continue
        t += 1
    return S @ E

def gpt_full_rank_decomposition(M):
    Q, R = np.linalg.qr(M.T)
    R *= Q[0,0] 
    Q /= Q[0,0]
    B, C = full_rank_decomposition(Q[:, 1:] @ R[1:, ])
    S = np.hstack([Q[:,0:1], B]).T
    E = np.vstack([R[0:1,:], C]).T
    return E, S 

def dual_halfspace_intersection(points):
    points = np.array([pt for pt in points if not np.allclose(pt, np.zeros(points.shape[1]))])
    n = len(points)
    halfspaces = np.vstack([np.hstack([-points, np.zeros(n).reshape(n,1)]),
                            np.hstack([points, -np.ones(n).reshape(n,1)])])
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1))
    c = np.zeros((halfspaces.shape[1],)); c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = -halfspaces[:, -1:]
    res = sc.optimize.linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    return HalfspaceIntersection(halfspaces, res.x[:-1])

def plot_convex_hull(hull, points=None, fill=True):
    points = hull.points if type(points) == type(None) else points
    d = points.shape[1]

    fig = plt.figure()
    if d == 3:
        ax = fig.add_subplot(111, projection="3d") 
    else:    
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')

    ax.plot(*[points[:,i] for i in range(d)], 'bo')

    if d == 3:
        if fill:
            for simplex in hull.simplices:
                f = a3.art3d.Poly3DCollection([[[points[simplex[i], j] for j in range(d)] for i in range(d)]])
                f.set_color(colors.rgb2hex(sc.rand(3)))
                f.set_edgecolor('k')
                f.set_alpha(0.4)
                ax.add_collection3d(f)
        else:
            for simplex in hull.simplices:
                ax.plot(*[points[simplex, i] for i in range(d)], 'black')
    else:
        for simplex in hull.simplices:
            ax.plot(*[points[simplex, i] for i in range(d)], 'black')
        plt.fill(*[points[hull.vertices,i] for i in range(d)], color=colors.rgb2hex(sc.rand(3)), alpha=0.3)

class GPT:
    @classmethod
    def from_probability_table(cls, P):
        effects, states = gpt_full_rank_decomposition(P)
        return GPT(effects, states)

    def __init__(self, effects, states, unit_effect=None, maximally_mixed_state=None):
        self.unit_effect = unit_effect if type(unit_effect) != type(None) else \
                           np.eye(states.shape[0])[0]
        self.maximally_mixed_state = maximally_mixed_state if type(maximally_mixed_state) != type(None) else \
                                     np.mean(states, axis=1)
        self.effects = np.vstack([effects, self.unit_effect - effects])
        self.states = states

        self.state_space = ConvexHull(self.states.T[:, 1:])
        self.effect_space = ConvexHull(self.effects)

        self.logical_states = dual_halfspace_intersection(self.effects)
        self.logical_effects = dual_halfspace_intersection(self.states.T)

        self.logical_state_space = ConvexHull(self.logical_states.intersections[:, 1:])
        self.logical_effect_space = ConvexHull(self.logical_effects.intersections)
    
        self.d = self.states.shape[0]

    def sample_measurement(self, m, max_iter=100):
        @jax.jit
        def unity(V):
            M = V.reshape(m, self.d)
            return jp.linalg.norm(jp.sum(M, axis=0) - self.unit_effect)**2

        @jax.jit
        def consistent_min(V):
            M = V.reshape(m, self.d)
            return (M @ self.states).flatten()

        @jax.jit
        def consistent_max(V):
            M = V.reshape(m, self.d)
            return -(M @ self.states).flatten() + 1

        i = 0
        while i < max_iter:
            V = np.random.randn(m*self.d)
            result = sc.optimize.minimize(unity, V,\
                                jac=jax.jit(jax.jacrev(unity)),\
                                tol=1e-16,\
                                constraints=[{"type": "ineq",\
                                             "fun": consistent_min,\
                                             "jac": jax.jit(jax.jacrev(consistent_min))},\
                                            {"type": "ineq",\
                                             "fun": consistent_max,\
                                             "jac": jax.jit(jax.jacrev(consistent_max))}],\
                                options={"maxiter": 5000},
                                method="SLSQP")
            M = result.x.reshape(m, self.d)
            if self.valid_measurement(M):
                return M
            i += 1
    
    def valid_measurement(self, M):
        return np.all(M @ self.states >= 0) and np.all(M @ self.states <= 1) and np.allclose(np.sum(M, axis=0), self.unit_effect)

    def sample_states(self, n, max_iter=100):
        @jax.jit
        def info_complete(V):
            S = V.reshape(self.d-1, n)
            S = jp.vstack([jp.ones(n).reshape(1, n), S])
            return (jp.linalg.matrix_rank(S) - self.d).astype(float)**2

        @jax.jit
        def consistent_min(V):
            S = V.reshape(self.d-1, n)
            S = jp.vstack([jp.ones(n).reshape(1, n), S])
            return (self.effects @ S).flatten()

        @jax.jit
        def consistent_max(V):
            S = V.reshape(self.d-1, n)
            S = jp.vstack([jp.ones(n).reshape(1, n), S])
            return -(self.effects @ S).flatten() + 1

        i = 0
        while i < max_iter:
            V = np.random.randn(n*(self.d-1))
            result = sc.optimize.minimize(info_complete, V,\
                                jac=jax.jit(jax.jacrev(info_complete)),\
                                tol=1e-16,\
                                constraints=[{"type": "ineq",\
                                             "fun": consistent_min,\
                                             "jac": jax.jit(jax.jacrev(consistent_min))},\
                                            {"type": "ineq",\
                                             "fun": consistent_max,\
                                             "jac": jax.jit(jax.jacrev(consistent_max))}],\
                                options={"maxiter": 5000},
                                method="SLSQP")
            S = result.x.reshape(self.d-1, n)
            S = np.vstack([np.ones(n).reshape(1,n), S])
            if self.valid_states(S):
                return S
            i += 1
    
    def valid_states(self, S):
        return np.all(self.effects @ S >= 0) and np.all(self.effects @ S <= 1)

    def minimize_quantumness(self, n, p=2, max_iter=100):
        m = n

        @partial(jax.jit, static_argnums=(1,2,3))
        def jit_quantumness(P, n, d, p):
            a, A, B = [1], [P], []
            for i in range(1, n+1):
                a.append(A[-1].trace()/i)
                B.append(A[-1] - a[-1]*jp.eye(n))
                A.append(P @ B[-1])
            j = n - d
            Phi = sum([((-1 if i == 0 else 1)*a[n-j-1]*a[i]/a[n-j]**2 + \
                         (i if i < 2 else -1)*a[i-1]/a[n-j])*\
                            jp.linalg.matrix_power(P, n-j-i)
                                for i in range(d)])
            S = jp.linalg.svd(np.eye(n) - Phi, compute_uv=False)
            return jp.sum(S**p)**(1/p) if p != np.inf else jp.max(S)

        @jax.jit
        def jit_wrapped_quantumness(V):
            M = V[:m*self.d:].reshape(m, self.d)
            S = V[m*self.d:].reshape(self.d-1, n)
            S = jp.vstack([jp.ones(n).reshape(1, n), S])
            return jit_quantumness(M @ S, n, self.d, p)

        @jax.jit
        def quantumness(V):
            M = V[:m*self.d:].reshape(m, self.d)
            S = V[m*self.d:].reshape(self.d-1, n)
            S = jp.vstack([jp.ones(n).reshape(1, n), S])
            Phi = jp.linalg.pinv(M @ S)
            sing_vals = jp.linalg.svd(np.eye(n) - Phi, compute_uv=False)
            return jp.sum(sing_vals**p)**(1/p) if p != np.inf else jp.max(S)

        @jax.jit
        def effects_unity(V):
            M = V[:m*self.d].reshape(m, self.d)
            return jp.linalg.norm(jp.sum(M, axis=0) - self.unit_effect)**2

        @jax.jit
        def effects_consistent_min(V):
            M = V[:m*self.d].reshape(m, self.d)
            return (M @ self.states).flatten()

        @jax.jit
        def effects_consistent_max(V):
            M = V[:m*self.d].reshape(m, self.d)
            return -(M @ self.states).flatten() + 1

        @jax.jit
        def states_consistent_min(V):
            S = V[m*self.d:].reshape(self.d-1, n)
            S = jp.vstack([jp.ones(n).reshape(1, n), S])
            return (self.effects @ S).flatten()

        @jax.jit
        def states_consistent_max(V):
            S = V[m*self.d:].reshape(self.d-1, n)
            S = jp.vstack([jp.ones(n).reshape(1, n), S])
            return -(self.effects @ S).flatten() + 1

        i = 0
        while i < max_iter:
            V = np.random.randn(m*self.d + n*(self.d-1))
            result = sc.optimize.minimize(quantumness, V,\
                                jac=jax.jit(jax.jacrev(quantumness)),\
                                tol=1e-16,\
                                constraints=[{"type": "eq",\
                                              "fun": effects_unity,\
                                              "jac": jax.jit(jax.jacrev(effects_unity))},\
                                            {"type": "ineq",\
                                              "fun": effects_consistent_min,\
                                              "jac": jax.jit(jax.jacrev(effects_consistent_min))},\
                                            {"type": "ineq",\
                                              "fun": effects_consistent_max,\
                                              "jac": jax.jit(jax.jacrev(effects_consistent_max))},\
                                            {"type": "ineq",\
                                              "fun": states_consistent_min,\
                                              "jac": jax.jit(jax.jacrev(states_consistent_min))},\
                                             {"type": "ineq",\
                                              "fun": states_consistent_max,\
                                              "jac": jax.jit(jax.jacrev(states_consistent_max))}],\
                                options={"maxiter": 5000},
                                method="trust-constr")
            M = result.x[:m*self.d].reshape(m, self.d)
            S = result.x[m*self.d:].reshape(self.d-1, n)
            S = np.vstack([np.ones(n).reshape(1,n), S])
            if self.valid_measurement(M) and self.valid_states(S) and \
                    np.all(M @ S >=0) and np.all(M @ S <= 1): 
                return M, S
            i += 1

#BoxWorld = GPT(np.array([[1,0,1], [-1,0,1], [0,1,1], [0,-1,1]])/2,\
#               np.array([[-1,-1,1], [-1,1,1], [1,1,1], [1,-1,1]]).T,\
#               unit_effect=np.array([0,0,1]))

class GPTStates:
    def __init__(self, states):
        self.states = states

class GPTEffects:
    def __init__(self, effects):
        self.effects = effects
        self.inverted = False

    def __lshift__(self, other):
        if np.all(other >= 0) and np.all(other <= 1) and np.isclose(np.sum(other),1):
            return np.linalg.pinv(self.effects) @ other
        return self.effects @ other

    def __invert__(self):
        self.inverted = True if not self.inverted else False
        return self

    def __inverted__(self, A):
        A = A if not self.inverted else spectral_inverse(A)
        self.inverted = False if self.inverted else self.inverted
        return A

    def __or__(self, other):
        if type(other) == GPTStates:
            return self.__inverted__(self.effects @ other.states)