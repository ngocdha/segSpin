import numpy as np
from numpy.linalg import eigh
from functools import reduce
import matplotlib.pyplot as plt

# Pauli and helpers (same as your code)
sx = np.array([[0,1],[1,0]], dtype=np.complex128)
sy = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
sz = np.array([[1,0],[0,-1]], dtype=np.complex128)
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

# Your z-coupling image Hamiltonians
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

# ---- Low-energy Gibbs approximation from the lowest k states ----
def lowT_gibbs_obs(H, beta, O_ops, k):
    # Small-system demo: use dense eigendecomposition and then truncate.
    evals, U = eigh(H)
    idx = np.argsort(evals)[:k]
    E = evals[idx]
    V = U[:, idx]       # columns are |E_m>
    w = np.exp(-beta*E)
    Zk = float(np.sum(w))
    weights = w / Zk

    # diagonal expectations in the truncated set
    # <E_m|O|E_m> for each O in O_ops
    obs = []
    for O in O_ops:
        vOv = np.einsum('im,ij,jm->m', V.conj(), O, V)  # size k
        obs.append(float(np.dot(weights, vOv.real)))
    return obs, (E, weights)

# ---- Example usage (swap in DMRG on big systems) ----
if __name__ == "__main__":
    # 1×8 toy
    img1d = np.concatenate([np.zeros(4), 255*np.ones(4)])
    beta = 2.0
    mu = 1.0
    seeds = [(0, +1, 5.0)]   # z-axis labels
    k = 6                    # keep 6 lowest states

    n, nrows, ncols, edges = build_Jz_from_image(img1d, kappa=4.0)
    H_I  = build_HI(n, edges)
    H_l  = build_Hl(n, seeds)
    Hfull = H_I + mu*H_l

    # Build the list of site operators O = {S_i^z}
    O_sites = [op_on(n, i, Sz) for i in range(n)]

    # Approximate <S_i^z> at temperature beta using truncated spectrum of Hfull
    sz_approx = []
    for i in range(n):
        val, meta = lowT_gibbs_obs(Hfull, beta, [O_sites[i]], k)
        sz_approx.append(val[0])
    sz_approx = np.array(sz_approx)
    print("<S_i^z> (low-energy approx):", sz_approx)

    l_site = 0
    O_pairs = [op2_on(n, l_site, i, Sz, Sz) for i in range(n)]
    szz_approx = []
    for i in range(n):
        val, _ = lowT_gibbs_obs(H_I, beta, [O_pairs[i]], k)
        szz_approx.append(val[0])
    szz_approx = np.array(szz_approx)
    print(f"<S_{l_site}^z S_i^z> (low-energy approx):", szz_approx)

