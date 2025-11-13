import numpy as np
from numpy.linalg import eigh
from functools import reduce

# Pauli spin-1/2
sx = np.array([[0, 1],[1, 0]], dtype=np.complex128)
sy = np.array([[0, -1j],[1j, 0]], dtype=np.complex128)
sz = np.array([[1, 0],[0,-1]], dtype=np.complex128)
Sx, Sy, Sz = 0.5*sx, 0.5*sy, 0.5*sz
I2 = np.eye(2, dtype=np.complex128)

def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def op_on(n, site, A):
    mats = [I2]*n
    mats[site] = A
    return kron_all(mats)

def op2_on(n, i, j, A, B):
    mats = [I2]*n
    mats[i] = A
    mats[j] = B
    return kron_all(mats)

def build_Jz_from_image(img, kappa=4.0):
    arr = img.astype(float).copy()
    if arr.ndim == 1:
        nrows, ncols = 1, arr.shape[0]
        grid = arr.reshape(1, -1)
    else:
        nrows, ncols = arr.shape
        grid = arr
    sigma = grid.std(ddof=0) if grid.std(ddof=0) > 0 else 1.0
    edges = []
    for r in range(nrows):
        for c in range(ncols):
            i = r*ncols + c
            if c+1 < ncols:
                diff = abs(grid[r,c] - grid[r,c+1])
                Jz = 2.0 - (kappa/sigma)*diff
                edges.append((i, i+1, Jz))
            if r+1 < nrows:
                diff = abs(grid[r,c] - grid[r+1,c])
                Jz = 2.0 - (kappa/sigma)*diff
                j = (r+1)*ncols + c
                edges.append((i, j, Jz))
    n = nrows*ncols
    return n, nrows, ncols, edges

def build_HI(n, edges):
    H_I = np.zeros((2**n, 2**n), dtype=np.complex128)
    for (i, j, Jz_ij) in edges:
        H_I -= Jz_ij * op2_on(n, i, j, Sz, Sz)
    return H_I

def build_Hl(n, seeds):
    H_l = np.zeros((2**n, 2**n), dtype=np.complex128)
    if seeds:
        for (site, label, strength) in seeds:
            H_l -= (strength * label) * op_on(n, site, Sz)
    return H_l

# exact
def gibbs_from_H_exact(H, beta, return_eigs=False):
    evals, U = eigh(H)
    w = np.exp(-beta*evals)
    Z = float(np.sum(w))
    rho = U @ np.diag(w/Z) @ U.conj().T
    if return_eigs:
        return rho, Z, evals, U
    return rho, Z, evals

# truncated (k lowest)
def gibbs_from_H_trunc(H, beta, k, solver='dense', return_eigs=False):
    if solver == 'sparse':
        try:
            from scipy.sparse.linalg import eigsh
            E, V = eigsh(H, k=k, which='SA')  # lowest
            idx = np.argsort(E)
            E, V = E[idx], V[:, idx]
        except Exception:
            evals, U = eigh(H)
            E, V = evals[:k], U[:, :k]
    else:
        evals, U = eigh(H)
        E, V = evals[:k], U[:, :k]
    w = np.exp(-beta*E)
    Zk = float(np.sum(w))
    rho_k = (V * (w/Zk)) @ V.conj().T
    if return_eigs:
        return rho_k, Zk, E, V
    return rho_k, Zk, E

# unified
def gibbs_from_H(H, beta, mode='exact', k=None, solver='dense', return_eigs=False):
    if mode == 'trunc':
        assert k is not None and k >= 1
        return gibbs_from_H_trunc(H, beta, k, solver=solver, return_eigs=return_eigs)
    return gibbs_from_H_exact(H, beta, return_eigs=return_eigs)

def stats_Sz(img, beta, mu, seeds, kappa=4.0, label_site=0,
             use_trunc=False, k=None, solver='dense'):
    n, nrows, ncols, edges = build_Jz_from_image(img, kappa=kappa)
    H_I = build_HI(n, edges)
    H_l = build_Hl(n, seeds)
    H_full = H_I + mu*H_l

    mode_full = 'trunc' if use_trunc else 'exact'
    mode_I    = 'trunc' if use_trunc else 'exact'

    rho_full, Z_full, _ = gibbs_from_H(H_full, beta, mode=mode_full, k=k, solver=solver)
    rho_I,    Z_I,    _ = gibbs_from_H(H_I,     beta, mode=mode_I,    k=k, solver=solver)

    Sz_list = [op_on(n, i, Sz) for i in range(n)]
    Sz_expect = np.array([np.real(np.trace(rho_full @ Sz_list[i])) for i in range(n)])

    Sz_label = Sz_list[label_site]
    SzSz_label = np.array([np.real(np.trace(rho_I @ (Sz_label @ Sz_list[i])))
                           for i in range(n)])

    return {
        "H_I": H_I, "H_l": H_l, "H_full": H_full,
        "rho_full": rho_full, "rho_I": rho_I,
        "Z_full": Z_full, "Z_I": Z_I,
        "Sz_expect": Sz_expect,
        "SzSz_label": SzSz_label,
        "n": n, "nrows": nrows, "ncols": ncols
    }

# example
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img1d = np.concatenate([np.zeros(5), 255*np.ones(5)])
    beta = 2.0
    mu = 0.0
    seeds = [(0, +1, 5.0)]
    label_site = 0

    # exact
    out = stats_Sz(img1d, beta, mu, seeds, kappa=4.0, label_site=label_site,
                   use_trunc=False)

    # truncated (e.g., keep 6 lowest)
    out_tr = stats_Sz(img1d, beta, mu, seeds, kappa=4.0, label_site=label_site,
                      use_trunc=True, k=20, solver='dense')

    Cz  = out["Sz_expect"];     Czz  = 2.0 * out["SzSz_label"];     n  = out["n"]
    CzT = out_tr["Sz_expect"];  CzzT = 2.0 * out_tr["SzSz_label"]

    plt.figure(figsize=(9,4))
    plt.plot(range(n), Cz,  'o-',  label='<S_i^z> exact')
    plt.plot(range(n), Czz, 's--', label='2<S_l^z S_i^z> exact')
    plt.plot(range(n), CzT,  'o-',  alpha=0.6, label='<S_i^z> trunc k=20')
    plt.plot(range(n), CzzT, 's--', alpha=0.6, label='2<S_l^z S_i^z> trunc k=20')
    plt.xlabel('Pixel index i'); plt.ylabel('Value'); plt.legend(); plt.tight_layout(); plt.show()
