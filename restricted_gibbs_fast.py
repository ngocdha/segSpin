import numpy as np
from itertools import combinations
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def build_Jz_edges_2d(img, kappa=4.0):
    img = np.asarray(img, dtype=float)
    H, W = img.shape
    sigma = img.std(ddof=0)
    sigma = sigma if sigma > 0 else 1.0

    edges = []
    for r in range(H):
        for c in range(W):
            i = r * W + c
            if c + 1 < W:
                diff = abs(img[r, c] - img[r, c + 1])
                Jz = 2.0 - (kappa / sigma) * diff
                edges.append((i, i + 1, float(Jz)))
            if r + 1 < H:
                diff = abs(img[r, c] - img[r + 1, c])
                Jz = 2.0 - (kappa / sigma) * diff
                j = (r + 1) * W + c
                edges.append((i, j, float(Jz)))

    return H * W, H, W, edges


def sz_from_occ(occ_bool):
    # occupied = up => +1/2, empty = down => -1/2
    return np.where(occ_bool, 0.5, -0.5)


def build_basis_k(n, k):
    if k == 0:
        return [tuple()]
    return list(combinations(range(n), k))


def build_xxz_sector_sparse(n, edges, k):
    """
    Build sparse XXZ Hamiltonian in the fixed N_up=k sector using k-particle tuples.
    H = -sum_edges [(SxSx+SySy) + Jz SzSz]
    XY part => hop one particle across an edge with amplitude -1/2.
    ZZ part => diagonal energy from SzSz.
    """
    basis = build_basis_k(n, k)
    dim = len(basis)
    index = {state: i for i, state in enumerate(basis)}

    # neighbor list for hopping
    nbrs = [[] for _ in range(n)]
    for (u, v, _) in edges:
        nbrs[u].append(v)
        nbrs[v].append(u)

    # Precompute diagonal Sz values for each basis state
    # occ[a, i] = True if site i occupied in basis state a
    occ = np.zeros((dim, n), dtype=bool)
    for a, st in enumerate(basis):
        for site in st:
            occ[a, site] = True
    sz = sz_from_occ(occ)  # shape (dim, n)

    # Diagonal ZZ energy
    diag = np.zeros(dim, dtype=np.float64)
    for (i, j, Jz) in edges:
        diag += -Jz * (sz[:, i] * sz[:, j])

    # Off-diagonal XY hopping (COO triplets)
    rows, cols, data = [], [], []
    # add diagonal
    rows.extend(range(dim))
    cols.extend(range(dim))
    data.extend(diag.tolist())

    if k > 0:
        for a, st in enumerate(basis):
            st_set = set(st)
            for p_idx, p in enumerate(st):
                for q in nbrs[p]:
                    if q in st_set:
                        continue  # cannot hop into occupied site
                    new_sites = list(st)
                    new_sites[p_idx] = q
                    new_sites.sort()
                    new_state = tuple(new_sites)
                    b = index.get(new_state, None)
                    if b is not None:
                        rows.append(a)
                        cols.append(b)
                        data.append(-0.5)

    Hk = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()
    return Hk, basis, occ


def hutchinson_szsz_sector(Hk, occ, beta, label_site=0, n_samples=64, seed=0):
    """
    Estimate:
      Z = Tr(exp(-beta Hk))
      N_i = Tr(exp(-beta Hk) * Sz_label * Sz_i)
    using Hutchinson estimator with expm_multiply.

    Sz_label*Sz_i is diagonal in this basis.
    """
    rng = np.random.default_rng(seed)
    dim, n = occ.shape

    sz = sz_from_occ(occ)          # (dim, n)
    szl = sz[:, label_site]        # (dim,)
    O_diag = szl[:, None] * sz     # (dim, n)

    Z_acc = 0.0
    N_acc = np.zeros(n, dtype=np.float64)

    for _ in range(n_samples):
        v = rng.integers(0, 2, size=dim) * 2 - 1
        v = v.astype(np.float64)

        w = spla.expm_multiply((-beta) * Hk, v)  # w = exp(-beta Hk) v

        Z_acc += np.dot(v, w)

        tmp = v * w
        N_acc += (O_diag.T @ tmp)

    Z_est = Z_acc / n_samples
    N_est = N_acc / n_samples
    corr = N_est / Z_est  # <Sz_label Sz_i>
    return Z_est, corr


def correlations_upto_k(img, beta, kappa, k_max, label_site=0, n_samples=64, seed=0):
    """
    Compute correlations in the truncated space (direct sum of k=0..k_max sectors):
      <Sz_label Sz_i>_trunc = sum_k Z_k * <...>_k / sum_k Z_k
    """
    n, H, W, edges = build_Jz_edges_2d(img, kappa=kappa)

    Z_total = 0.0
    Num_total = np.zeros(n, dtype=np.float64)

    for k in range(k_max + 1):
        Hk, basis_k, occ_k = build_xxz_sector_sparse(n, edges, k)
        Zk, corr_k = hutchinson_szsz_sector(
            Hk, occ_k, beta, label_site=label_site, n_samples=n_samples, seed=seed + 101 * k
        )
        Z_total += Zk
        Num_total += Zk * corr_k

    corr_trunc = Num_total / Z_total  # <Sz_label Sz_i> in truncated mixture
    C_map = (2.0 * corr_trunc).reshape(H, W)
    return C_map


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(8, 8)).astype(float)

    beta = 2.0
    kappa = 3.0
    k_max = 3
    label_site = 0
    n_samples = 64   # increase for smoother estimates

    C_map = correlations_upto_k(
        img, beta=beta, kappa=kappa, k_max=k_max, label_site=label_site, n_samples=n_samples, seed=0
    )

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    plt.title("Image (6×6)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(C_map, interpolation="nearest")
    plt.title(f"Upto N_up≤{k_max}: 2<Sz_l Sz_i>\n(beta={beta}, samples={n_samples})")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
