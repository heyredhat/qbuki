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
        E_ = V[:n_elements*n_dim].reshape(n_elements, n_dim)
        S_ = V[n_elements*n_dim:].reshape(n_dim-1, n_elements)
        S_ = jp.vstack([jp.ones(n_elements).reshape(1, n_elements), S_])
        return jit_quantumness(E_ @ S_, n_elements, n_dim, p)