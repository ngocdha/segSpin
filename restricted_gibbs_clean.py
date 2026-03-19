import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.sparse as sp
import scipy.linalg as la
from sklearn.datasets import load_digits

# ============================================================
# Utilities
# ============================================================

def sz_from_bit(bit: int) -> float:
    return 0.5 if bit else -0.5


# ============================================================
# Image constructors / loaders
# ============================================================

def make_plus_image(H=5, W=5, white=255.0, black=0.0, thickness=1):
    img = np.full((H, W), white, dtype=float)
    r0 = H // 2
    c0 = W // 2
    for t in range(-(thickness // 2), thickness - (thickness // 2)):
        rr = int(np.clip(r0 + t, 0, H - 1))
        cc = int(np.clip(c0 + t, 0, W - 1))
        img[rr, :] = black
        img[:, cc] = black
    return img

def make_zero_image_8x8(white=255.0, black=0.0):
    """
    8x8 handmade hollow zero:
      - outer background white
      - 6x6 black ring in rows/cols 1..6
      - 4x4 white hole in rows/cols 2..5
    """
    img = np.full((8, 8), white, dtype=float)
    img[1:7, 1:7] = black
    img[2:6, 2:6] = white
    return img

def load_digits_8x8(index=0, invert=False):
    """
    Built-in sklearn handwritten digits dataset, already 8x8.
    No internet/download required.
    """
    digits = load_digits()
    img = digits.images[index].astype(float)

    img = img - img.min()
    if img.max() > 0:
        img = 255.0 * img / img.max()

    if invert:
        img = 255.0 - img

    return img.astype(float)


# ============================================================
# Graph construction
# ============================================================

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


# ============================================================
# Sector basis + XXZ Hamiltonian
# ============================================================

def basis_bitmasks_k(n, k):
    if k < 0 or k > n:
        return []
    if k == 0:
        return [0]

    out = []
    for sites in combinations(range(n), k):
        s = 0
        for i in sites:
            s |= (1 << i)
        out.append(s)
    return out

def build_xxz_sector_sparse_bitbasis(n, edges, k):
    """
    Exact sector Hamiltonian for the active-label, no-pruning baseline.

    H = sum_{<i,j>}
            [ -J_ij Sz_i Sz_j
              - 1/2 (S_i^+ S_j^- + S_i^- S_j^+) ]

    restricted to the subspace with exactly k up spins.
    """
    basis = basis_bitmasks_k(n, k)
    dim = len(basis)

    if dim == 0:
        return sp.csr_matrix((0, 0), dtype=np.float64), basis

    idx = {s: a for a, s in enumerate(basis)}

    rows = []
    cols = []
    data = []

    # Diagonal part: ZZ energy
    for a, s in enumerate(basis):
        e = 0.0
        for (i, j, Jz) in edges:
            bi = (s >> i) & 1
            bj = (s >> j) & 1
            e += -Jz * sz_from_bit(bi) * sz_from_bit(bj)

        rows.append(a)
        cols.append(a)
        data.append(e)

    # Off-diagonal part: XY spin exchange
    if k > 0:
        for a, s in enumerate(basis):
            for (i, j, _) in edges:
                bi = (s >> i) & 1
                bj = (s >> j) & 1

                if bi != bj:
                    # flip bits i and j
                    t = s ^ ((1 << i) | (1 << j))
                    b = idx.get(t, None)
                    if b is not None:
                        rows.append(a)
                        cols.append(b)
                        data.append(-0.5)

    Hk = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()
    return Hk, basis

# ============================================================

def build_sz_table(basis, n):
    """
    sz[a, i] = spin value (+/- 1/2) at site i in basis state a
    """
    dim = len(basis)
    sz = np.empty((dim, n), dtype=np.float64)
    for a, s in enumerate(basis):
        for i in range(n):
            sz[a, i] = sz_from_bit((s >> i) & 1)
    return sz

def exact_sector_observables_trace(Hk, basis, beta, n, label_site):
    """
    Compute observables from the exact trace formula:

        Z_k = Tr(exp(-beta H_k))
        <O>_k = Tr(O exp(-beta H_k)) / Z_k

    using explicit matrix exponential in the sector basis.

    Returns
    -------
    Zk : float
    corr_k : ndarray, shape (n,)
        corr_k[i] = <Sz_label Sz_i>_k
    mag_k : ndarray, shape (n,)
        mag_k[i] = <Sz_i>_k
    """
    dim = len(basis)

    if dim == 0:
        return 0.0, np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)

    Hd = Hk.toarray()
    Hd = 0.5 * (Hd + Hd.T)  # exact symmetry enforcement for numerical safety

    rho = la.expm(-beta * Hd)
    Zk = float(np.trace(rho))

    sz = build_sz_table(basis, n)
    ls = int(label_site)
    if not (0 <= ls < n):
        raise ValueError("label_site out of range")

    mag_k = np.zeros(n, dtype=np.float64)
    corr_k = np.zeros(n, dtype=np.float64)

    # Observables are diagonal in the computational basis
    for i in range(n):
        O_mag_diag = sz[:, i]
        O_corr_diag = sz[:, ls] * sz[:, i]

        O_mag = np.diag(O_mag_diag)
        O_corr = np.diag(O_corr_diag)

        mag_k[i] = float(np.trace(O_mag @ rho) / Zk)
        corr_k[i] = float(np.trace(O_corr @ rho) / Zk)

    return Zk, corr_k, mag_k


# ============================================================
# Plotting
# ============================================================

def save_correlation_plot(img, C_map, beta, kappa, k, outpath,
                          title_prefix="Up to", label_site=None):
    H, W = img.shape

    plt.figure(figsize=(10, 4))

    ax0 = plt.subplot(1, 2, 1)
    ax0.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax0.set_title("Image")
    ax0.set_xticks([])
    ax0.set_yticks([])

    if label_site is not None:
        rr = label_site // W
        cc = label_site % W
        ax0.scatter([cc], [rr], s=120, marker="*", c="yellow",
                    edgecolors="black", linewidths=0.8)

    ax1 = plt.subplot(1, 2, 2)
    im = ax1.imshow(C_map, interpolation="nearest")
    ax1.set_title(f"{title_prefix} k={k}: 2<Sz_l Sz_i>\n(beta={beta}, kappa={kappa})")
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()

def save_classification_plot(img, cls_map, outpath, title="Final classification", label_site=None):
    H, W = img.shape

    plt.figure(figsize=(10, 4))

    ax0 = plt.subplot(1, 2, 1)
    ax0.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax0.set_title("Image")
    ax0.set_xticks([])
    ax0.set_yticks([])

    if label_site is not None:
        rr = label_site // W
        cc = label_site % W
        ax0.scatter([cc], [rr], s=120, marker="*", c="yellow",
                    edgecolors="black", linewidths=0.8)

    ax1 = plt.subplot(1, 2, 2)
    ax1.imshow(cls_map, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax1.set_title(title)
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()

def classify_from_C(C_map, tau_class=0.0, pos_value=255, neg_value=0, unk_value=127):
    C = np.asarray(C_map, dtype=np.float64)

    if tau_class <= 0.0:
        return np.where(C >= 0.0, pos_value, neg_value).astype(np.uint8)

    out = np.full(C.shape, unk_value, dtype=np.uint8)
    out[C >= +tau_class] = pos_value
    out[C <= -tau_class] = neg_value
    return out


# ============================================================
# Clean exact active-label restricted Gibbs runner
# ============================================================

def run_exact_active_label(
    img,
    beta,
    kappa,
    k_top,
    label_site=0,
    out_dir="out_exact_active_label",
    save_every_k=True
):
    """
    Exact restricted Gibbs baseline with ACTIVE label and NO pruning.

    Observable:
        C_i = 2 <Sz_label Sz_i>

    Sector accumulation:
        <O>_{<=k_top} = [sum_{k=0}^{k_top} Z_k <O>_k] / [sum_{k=0}^{k_top} Z_k]
    """
    img = np.asarray(img, dtype=float)
    H, W = img.shape
    n = H * W

    if not (0 <= label_site < n):
        raise ValueError("label_site out of range")

    _, _, _, edges = build_Jz_edges_2d(img, kappa=kappa)

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    Z_cum = 0.0
    Num_cum = np.zeros(n, dtype=np.float64)
    Mag_cum = np.zeros(n, dtype=np.float64)

    print(f"Image size: {H}x{W}, n={n}")
    print(f"beta={beta}, kappa={kappa}, k_top={k_top}, label_site={label_site}")

    for k in range(k_top + 1):
        print(f"\n=== k = {k} ===")

        Hk, basis = build_xxz_sector_sparse_bitbasis(n, edges, k)
        dimk = Hk.shape[0]
        print(f"sector dimension = {dimk}")

        if dimk == 0:
            Zk = 0.0
            corr_k = np.zeros(n, dtype=np.float64)
            mag_k = np.zeros(n, dtype=np.float64)
        else:
            Zk, corr_k, mag_k = exact_sector_observables_trace(
                Hk, basis, beta=beta, n=n, label_site=label_site
            )

        print(f"Zk = {Zk:.12g}")

        Z_cum += Zk
        Num_cum += Zk * corr_k
        Mag_cum += Zk * mag_k

        corr_upto = Num_cum / Z_cum
        mag_upto = Mag_cum / Z_cum
        C_now = 2.0 * corr_upto

        print(f"C(label,label) = {C_now[label_site]:.12g}")
        print(f"<Sz_label> = {mag_upto[label_site]:.12g}")

        if save_every_k:
            C_map = C_now.reshape(H, W)
            outpath = os.path.join(plots_dir, f"corr_upto_{k}.png")
            save_correlation_plot(
                img, C_map, beta, kappa, k, outpath,
                title_prefix="Up to", label_site=label_site
            )

    corr_final = Num_cum / Z_cum
    C_final = 2.0 * corr_final
    C_final_map = C_final.reshape(H, W)

    final_corr_path = os.path.join(out_dir, "final_correlation_map.png")
    save_correlation_plot(
        img, C_final_map, beta, kappa, "final", final_corr_path,
        title_prefix="Final", label_site=label_site
    )

    TAU_CLASS = 0.0
    cls_final = classify_from_C(C_final_map, tau_class=TAU_CLASS,
                                pos_value=255, neg_value=0, unk_value=127)

    final_cls_path = os.path.join(out_dir, "final_classification_map.png")
    save_classification_plot(
        img, cls_final, final_cls_path,
        title=f"Final classification (tau_class={TAU_CLASS})",
        label_site=label_site
    )

    print("\nDone.")
    print("Final output directory:", out_dir)

    return {
        "C_final": C_final,
        "corr_final": corr_final,
        "mag_final": Mag_cum / Z_cum,
        "Z_total": Z_cum,
    }


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":


    # 1) Small toy plus
    img = make_plus_image(H=5, W=5, white=255.0, black=0.0, thickness=1)

    # 2) Handmade hollow zero
    # img = make_zero_image_8x8(white=255.0, black=0.0)

    # 3) Built-in 8x8 handwritten digit
    # img = load_digits_8x8(index=0, invert=False)

    H, W = img.shape

    beta = 10.0
    kappa = 3.0
    k_top = 4
    label_site = 0

    out_dir = (
        f"out_exact_active_label_"
        f"{H}x{W}_"
        f"beta{beta}_"
        f"kappa{kappa}_"
        f"ktop{k_top}_"
        f"label{label_site}"
    )

    run_exact_active_label(
        img=img,
        beta=beta,
        kappa=kappa,
        k_top=k_top,
        label_site=label_site,
        out_dir=out_dir,
        save_every_k=True
    )