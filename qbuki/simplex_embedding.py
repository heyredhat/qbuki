# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/SimplexEmbedding.ipynb.

# %% auto 0
__all__ = ['quick_plot_3d_pts', 'hilbert_to_gpt', 'example1_gpt', 'example2_gpt', 'boxworld_gpt', 'rand_classical_gpt',
           'rand_quantum_gpt', 'sic_gpt', 'mub_gpt', 'polygonal_states', 'dualize_states', 'polygonal_gpt', 'tinyfier',
           'bloch_transform', 'plot_halfspaces', 'interior_point', 'dualize_effects', 'simplicial_embedding',
           'simplex_embedding', 'first_pass_simplex_embedding']

# %% ../nbs/SimplexEmbedding.ipynb 9
def quick_plot_3d_pts(*P):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    c = ["r", "g", "b", "y"]
    for i, p in enumerate(P):
        ax.scatter(p[0], p[1], p[2], c=c[i])
    ax.scatter(0,0,0,c="black")
    plt.show()

# %% ../nbs/SimplexEmbedding.ipynb 33
# Here we use Gellmann matrices to convert quantum states and effects into Bloch vectors.
def hilbert_to_gpt(states, effects):
    d = states[0].shape[0]
    basis = gellmann_basis(d)
    to_gellmann = lambda O: np.array([(O@b).trace() for b in basis[::-1]])
    from_gellman = lambda o: sum([c*basis[d**2-i] for i, c in enumerate(o)])
    return np.array([to_gellmann(o) for o in states]).T.real,\
           np.array([to_gellmann(o) for o in effects]).real,\
                     to_gellmann(np.eye(d)).real,\
                     to_gellmann(np.eye(d)/d).real

# %% ../nbs/SimplexEmbedding.ipynb 34
def example1_gpt():
    Zup, Zdown = np.array([1,0]), np.array([0,1])
    Xup, Xdown = np.array([1,1])/np.sqrt(2), np.array([1,-1])/np.sqrt(2)
    s = [np.outer(s, s.conj()) for s in [Zup, Zdown, Xup, Xdown]]
    e = [_/2 for _ in s]
    return hilbert_to_gpt(s, e)

# %% ../nbs/SimplexEmbedding.ipynb 35
def example2_gpt():
    return np.array([[1,0,0,0],\
                     [0,1,0,0],\
                     [0,0,1,0],\
                     [0,0,0,1]]),\
           np.array([[1,1,0,0],
                     [0,1,1,0],\
                     [0,0,1,1],\
                     [1,0,0,1]])/2,\
           np.array([1,1,1,1]),\
           np.mean(S, axis=1)

# %% ../nbs/SimplexEmbedding.ipynb 36
def boxworld_gpt():
    return np.array([[1,1,0],\
                     [1,0,1],\
                     [1,-1,0],\
                     [1,0,-1]]).T,\
           np.array([[1,-1,-1],\
                     [1,1,-1],\
                     [1,1,1],\
                     [1,-1,1]])/4,\
           np.array([1,0,0]),\
           np.array([1,0,0])

# %% ../nbs/SimplexEmbedding.ipynb 37
def rand_classical_gpt(n, r):
    E,S = rand_stochastic(n, r), rand_stochastic(r, n)
    return S, E, np.ones(r), np.ones(r)/r

# %% ../nbs/SimplexEmbedding.ipynb 38
def rand_quantum_gpt(d, n):
    return hilbert_to_gpt([rand_dm(d) for i in range(n)], rand_povm(d,n))

# %% ../nbs/SimplexEmbedding.ipynb 39
def sic_gpt(d):
    return hilbert_to_gpt([e/e.trace() for e in sic_povm(d)], sic_povm(d))

# %% ../nbs/SimplexEmbedding.ipynb 40
def mub_gpt(d):
    mubs = prime_mubs(d)
    mub_vecs = [v for mub in mubs for v in mub]
    S = [np.outer(v, v.conj()) for v in mub_vecs]
    E = [s/len(mubs) for s in S]
    return hilbert_to_gpt(S, E)

# %% ../nbs/SimplexEmbedding.ipynb 41
def polygonal_states(n):
    w = np.exp(2*np.pi*1j/n)
    return np.array([[1, (w**i).real, (w**i).imag] for i in range(n)]).T

def dualize_states(S, backend="qhull"):
    if backend == "qhull":
        if S.shape[0] == 2:
            return np.array([[-np.min(S[1:]), 1],[np.max(S[1:]), -1]])
        hull = sc.spatial.ConvexHull(S.T[:,1:])
        eq = hull.equations
        A, b = eq[:,:-1], eq[:, -1]
        return -np.hstack([b.reshape(b.shape[0],1), A])
    elif backend == "cdd":
        C = cdd.Matrix(S.T, number_type="float")
        C.rep_type = cdd.RepType.GENERATOR
        return np.array(cdd.Polyhedron(C).get_inequalities())

def polygonal_gpt(n):
    S = polygonal_states(n)
    e = dualize_states(S, backend="cdd")
    E = e/np.sum(e, axis=0)[0]
    if n == 3: # problem with n=3
        E = S.T/n
    return S, E, np.eye(3)[0], np.eye(3)[0]

# %% ../nbs/SimplexEmbedding.ipynb 47
def tinyfier(X):
    U, D, V = np.linalg.svd(X)
    r = np.isclose(D, 0).argmax(axis=0)
    return U[:,:r if r != 0 else D.shape[0]].T

# %% ../nbs/SimplexEmbedding.ipynb 52
def bloch_transform(I, S):
    Iproj = np.outer(I,I)/(I@I)
    notIproj = np.eye(I.shape[0]) - Iproj
    U, D, V = np.linalg.svd(notIproj @ S)
    r = np.isclose(D, 0).argmax(axis=0)
    return np.vstack([I, (U[:,:r if r != 0 else D.shape[0]].T) @ notIproj])

# %% ../nbs/SimplexEmbedding.ipynb 65
def dualize_states(S, backend="cdd"):
    if backend == "qhull":
        if S.shape[0] == 2:
            return np.array([[-np.min(S[1:]), 1],[np.max(S[1:]), -1]])
        hull = sc.spatial.ConvexHull(S.T[:,1:])
        eq = hull.equations
        A, b = eq[:,:-1], eq[:, -1]
        return -np.hstack([b.reshape(-1,1), A])
    elif backend == "cdd":
        C = cdd.Matrix(S.T, number_type="float")
        C.rep_type = cdd.RepType.GENERATOR
        return np.array(cdd.Polyhedron(C).get_inequalities())

# %% ../nbs/SimplexEmbedding.ipynb 70
def plot_halfspaces(halfspaces, L=4):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    xlim, ylim = (-L, L), (-L, L)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    x = np.linspace(-L, L, 100)
    symbols = ['-', '+', 'x', '*']
    signs = [0, 0, -1, -1]
    fmt = {"color": None, "edgecolor": "b", "alpha": 0.5}
    for h, sym, sign in zip(halfspaces, symbols, signs):
        hlist = h.tolist()
        fmt["hatch"] = sym
        if h[1]== 0:
            ax.axvline(-h[2]/h[0], label='{}x+{}y+{}=0'.format(*hlist))
            xi = np.linspace(xlim[sign], -h[2]/h[0], 100)
            ax.fill_between(xi, ylim[0], ylim[1], **fmt)
        else:
            ax.plot(x, (-h[2]-h[0]*x)/h[1], label='{}x+{}y+{}=0'.format(*hlist))
            ax.fill_between(x, (-h[2]-h[0]*x)/h[1], ylim[sign], **fmt)
    #x, y = zip(*halfspaces.intersections)
    #ax.plot(x, y, 'o', markersize=8)  
    
def interior_point(halfspaces):
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1))
    c = np.zeros((halfspaces.shape[1],)); c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = -halfspaces[:, -1:]
    res = sc.optimize.linprog(c, A_ub=A, b_ub=b, bounds=(None, None), method='highs')
    if res.success:
        return res.x[:-1]
    else:
        return np.zeros((halfspaces.shape[1]-1,))

def dualize_effects(E, backend="cdd"):
    if backend == "qhull":
        if E.shape[1] == 2:
            b, a = E[:,0], E[:,1]
            greater = [-b[i]/a[i] for i in range(len(a)) if a[i] > 0]
            less = [-b[i]/a[i] for i in range(len(a)) if a[i] < 0]
            return np.array([[1, np.max(greater)] if len(greater) != 0 else [0,-1],\
                             [1, np.min(less)] if len(less) != 0 else [0,1]]).T
        hs = np.roll(-E, -1, axis=1)
        half = sc.spatial.HalfspaceIntersection(hs, interior_point(hs))
        intersections = half.intersections.T
        return np.vstack([np.ones((1, intersections.shape[1])), intersections])
    elif backend == "cdd":
        C = cdd.Matrix(E, number_type="float")
        C.rep_type = cdd.RepType.INEQUALITY
        return np.array(cdd.Polyhedron(C).get_generators()).T

# %% ../nbs/SimplexEmbedding.ipynb 80
def simplicial_embedding(S, E, IA, MA):
    p_, Phi_ = cp.Variable(nonneg=True),\
               cp.Variable(shape=(S.shape[1], E.shape[0]), nonneg=True)
    problem = cp.Problem(cp.Minimize(p_),\
               [p_*MA + (1-p_)*IA - S @ Phi_ @ E == 0])
    problem.solve()
    return p_.value, Phi_.value

# %% ../nbs/SimplexEmbedding.ipynb 92
def simplex_embedding(S, E, I, M, backend="cdd"):
    bigToSmallStates = tinyfier(S)
    bigToSmallEffects = tinyfier(E.T).T
    BS = bloch_transform(I @ bigToSmallStates.T, bigToSmallStates @ S)
    BE = bloch_transform((bigToSmallEffects.T @ I).T, (E @ bigToSmallEffects).T)

    EA = E @ bigToSmallEffects @ BE
    SA = BS @ bigToSmallStates @ S
    IA = np.linalg.inv(BE) @ bigToSmallEffects.T @ bigToSmallStates.T @ np.linalg.inv(BS)
    MA = np.linalg.inv(BE) @ bigToSmallEffects.T @ np.outer(M, M) @ bigToSmallStates.T @ np.linalg.inv(BS)

    SA_star = dualize_states(SA, backend=backend)
    EA_star = dualize_effects(EA, backend=backend)
    p, Phi = simplicial_embedding(EA_star, SA_star, IA, MA)

    tauE = bigToSmallEffects @ BE @ EA_star @ Phi
    tauS = SA_star @ BS @ bigToSmallStates
    sigmaS = np.array([tauS[i,:]*(I @ tauE[:,i]) for i in range(tauS.shape[0])\
                  if not np.isclose((I @ tauE[:,i]),0)])
    sigmaE = np.array([tauE[:,i]/(I @ tauE[:,i]) for i in range(tauE.shape[1])\
                  if not np.isclose((I @ tauE[:,i]),0)]).T

    PE, PS = E @ sigmaE, sigmaS @ S
    return p, PE, PS

# %% ../nbs/SimplexEmbedding.ipynb 94
def first_pass_simplex_embedding(S, E, I, M):
    p_, Phi_ = cp.Variable(nonneg=True),\
               cp.Variable(shape=(S.shape[1], E.shape[0]), nonneg=True)
    problem = cp.Problem(cp.Minimize(p_),\
               [p_*np.outer(I,I) + (1-p_)*np.eye(E.shape[1]) - S @ Phi_ @ E == 0])
    problem.solve()
    p, Phi = p_.value, Phi_.value
    return p, E @ S @ Phi, E @ S
