import numpy as np
import scipy as sc
from scipy.spatial import ConvexHull
from scipy.spatial import HalfspaceIntersection

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

    def sample_measurement(self, n, max_iter=100):
        @jax.jit
        def unity(V):
            M = V.reshape(n, self.d)
            return jp.linalg.norm(jp.sum(M, axis=0) - self.unit_effect)**2

        @jax.jit
        def consistent_min(V):
            M = V.reshape(n, self.d)
            return (M @ self.states).flatten()

        @jax.jit
        def consistent_max(V):
            M = V.reshape(n, self.d)
            return -(M @ self.states).flatten() + 1

        i = 0
        while i < max_iter:
            V = np.random.randn(n*self.d)
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
            M = result.x.reshape(n, self.d)
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

def rand_probability_table(m, n, k):
    P = np.random.uniform(low=0, high=1, size=(m, n))
    
    @jax.jit
    def obj(V):
        A = V[:m*k].reshape(m, k)
        B = V[m*k:].reshape(k, n)
        return jp.linalg.norm(A@B - P)

    @jax.jit
    def consistency_max(V):
        A = V[:m*k].reshape(m, k)
        B = V[m*k:].reshape(k, n)
        return -(A@B).flatten() +1

    V = np.random.randn(m*k + k*n)
    result = sc.optimize.minimize(obj, V,\
                                  jac=jax.jit(jax.jacrev(obj)),\
                                  tol=1e-16,\
                                  constraints=[{"type": "ineq",\
                                                "fun": consistency_max,\
                                                "jac": jax.jit(jax.jacrev(consistency_max))}],\
                                  options={"maxiter": 5000},
                                  method="SLSQP")
    A = result.x[:m*k].reshape(m, k)
    B = result.x[m*k:].reshape(k, n)
    if not (np.all(A @ B >= 0) and np.all(A @ B <= 1)):
        return rand_probability_table(m, n, k)
    else:
        return A @ B

def rand_quantum_probability_table(d, m, n, field="complex", r=1):
    effects = [np.eye(d)] + [rand_effect(d, r=r, field=field) for _ in range(m-1)]
    states = [rand_dm(d, r=r, field=field) for _ in range(n)]
    return np.array([[(e @ s).trace() for s in states] for e in effects]).real

# min_quantumness
# gpt box