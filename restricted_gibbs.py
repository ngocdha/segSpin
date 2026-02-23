import math
import numpy as np
import matplotlib.pyplot as plt


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


def sz_from_bit(bit):
    return 0.5 if bit else -0.5


def build_restricted_xxz_H(n, edges, k, mode="exact"):
    if mode not in ("exact", "upto"):
        raise ValueError('mode must be "exact" or "upto"')

    if mode == "exact":
        basis = [s for s in range(1 << n) if s.bit_count() == k]
    else:
        basis = [s for s in range(1 << n) if s.bit_count() <= k]

    dim = len(basis)
    idx = {s: a for a, s in enumerate(basis)}

    Hk = np.zeros((dim, dim), dtype=np.float64)

    for a, s in enumerate(basis):
        diag = 0.0
        for (i, j, Jz) in edges:
            bi = (s >> i) & 1
            bj = (s >> j) & 1
            diag += -Jz * sz_from_bit(bi) * sz_from_bit(bj)
        Hk[a, a] = diag

        for (i, j, _) in edges:
            bi = (s >> i) & 1
            bj = (s >> j) & 1
            if bi != bj:
                t = s ^ ((1 << i) | (1 << j))
                b = idx.get(t, None)
                if b is not None:
                    Hk[a, b] += -0.5

    return Hk, basis


def gibbs_rho_from_H(H, beta):
    evals, U = np.linalg.eigh(H)
    w = np.exp(-beta * evals)
    Z = float(w.sum())
    rho = (U * (w / Z)) @ U.T
    return rho, Z, evals


def szsz_label_from_subspace_rho_diag(basis, rho_diag, n, label_site=0):
    out = np.zeros(n, dtype=np.float64)
    for a, s in enumerate(basis):
        pa = rho_diag[a]
        szl = sz_from_bit((s >> label_site) & 1)
        for i in range(n):
            out[i] += pa * szl * sz_from_bit((s >> i) & 1)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(6, 6)).astype(float)

    beta = 10.0
    kappa = 3.0
    label_site = 0
    k = 2

    n, Hh, Ww, edges = build_Jz_edges_2d(img, kappa=kappa)

    dim_exact = math.comb(n, k)
    dim_upto = sum(math.comb(n, kk) for kk in range(k + 1))
    print(f"n={n}, exact dim={dim_exact}, upto dim={dim_upto}")

    print(f"Building exact restricted H (N_up = {k})...")
    H_exact, basis_exact = build_restricted_xxz_H(n, edges, k=k, mode="exact")
    print("Diagonalizing exact restricted H...")
    rho_exact, Z_exact, evals_exact = gibbs_rho_from_H(H_exact, beta)
    rho_exact_diag = np.real(np.diag(rho_exact))

    print(f"Building upto restricted H (N_up ≤ {k})...")
    H_upto, basis_upto = build_restricted_xxz_H(n, edges, k=k, mode="upto")
    print("Diagonalizing upto restricted H...")
    rho_upto, Z_upto, evals_upto = gibbs_rho_from_H(H_upto, beta)
    rho_upto_diag = np.real(np.diag(rho_upto))

    SzSz_exact = szsz_label_from_subspace_rho_diag(basis_exact, rho_exact_diag, n, label_site=label_site)
    SzSz_upto = szsz_label_from_subspace_rho_diag(basis_upto, rho_upto_diag, n, label_site=label_site)

    C_map_exact = (2.0 * SzSz_exact).reshape(Hh, Ww)
    C_map_upto = (2.0 * SzSz_upto).reshape(Hh, Ww)

    all_e = np.concatenate([evals_exact, evals_upto])
    bins = np.histogram_bin_edges(all_e, bins="fd")

    plt.figure(figsize=(14, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    plt.title("Original image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.hist(evals_exact, bins=bins, density=True, alpha=0.6, label=f"Exact N_up={k}")
    plt.hist(evals_upto, bins=bins, density=True, alpha=0.6, label=f"Upto N_up≤{k}")
    plt.title(f"Eigenvalue distributions (beta={beta})")
    plt.xlabel("eigenvalue")
    plt.ylabel("density")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.imshow(C_map_exact, interpolation="nearest")
    plt.title(f"Exact sector: N_up={k}\n2<Sz_l Sz_i> (l={label_site}, beta={beta})")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(C_map_upto, interpolation="nearest")
    plt.title(f"Upto sector: N_up≤{k}\n2<Sz_l Sz_i> (l={label_site}, beta={beta})")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
