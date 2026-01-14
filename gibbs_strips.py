import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt

# Pauli spin-1/2 and identities
sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
Sx, Sy, Sz = 0.5 * sx, 0.5 * sy, 0.5 * sz
I2 = np.eye(2, dtype=np.complex128)


# Dense operator utilities

def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def op_on(n, site, A):
    mats = [I2] * n
    mats[site] = A
    return kron_all(mats)

def op2_on(n, i, j, A, B):
    mats = [I2] * n
    mats[i] = A
    mats[j] = B
    return kron_all(mats)



# 1D couplings from a row

def build_Jz_from_row(row, kappa=4.0):
    """
    Build nearest-neighbor edges for a 1D chain from a single image row.
    row: shape (W,)
    edges: list of (i, i+1, Jz_i)
    """
    arr = row.astype(float).copy()
    sigma = arr.std(ddof=0) if arr.std(ddof=0) > 0 else 1.0

    W = arr.shape[0]
    edges = []
    for i in range(W - 1):
        diff = abs(arr[i] - arr[i + 1])
        Jz = 2.0 - (kappa / sigma) * diff
        edges.append((i, i + 1, Jz))

    n = W
    return n, edges



# Hamiltonians and Gibbs

def build_HI_1d(n, edges):
    """
    H_I = - sum_{(i,i+1)} Jz_ij Sz_i Sz_{i+1}
    (dense 2^n x 2^n)
    """
    H_I = np.zeros((2**n, 2**n), dtype=np.complex128)
    for (i, j, Jz_ij) in edges:
        H_I -= Jz_ij * op2_on(n, i, j, Sz, Sz)
    return H_I

def build_Hl(n, seeds):
    """
    H_l = - sum seeds (strength * label) Sz_site
    seeds: list of (site, label, strength), label in {+1,-1}
    """
    H_l = np.zeros((2**n, 2**n), dtype=np.complex128)
    for (site, label, strength) in seeds:
        H_l -= (strength * label) * op_on(n, site, Sz)
    return H_l

def gibbs_from_H(H, beta):
    evals, U = eigh(H)
    w = np.exp(-beta * evals)
    Z = float(np.sum(w))
    rho = U @ np.diag(w / Z) @ U.conj().T
    return rho, Z, evals



# Row-level correlations

def row_SzSz_label_exact(row, beta, kappa=4.0, label_site=0,
                         pin_left_black=True,
                         pin_strength=20.0,
                         mu=1.0,
                         black_label=+1):
    """
    Exact Gibbs on a 1D row; return Czz[i] = 2 * <Sz_label Sz_i> under rho.

    If pin_left_black=True, we add a strong seed at site 0 to bias the constraint.
    Convention: black_label=+1 means "black corresponds to +Sz".
    """
    n, edges = build_Jz_from_row(row, kappa=kappa)
    H_I = build_HI_1d(n, edges)

    seeds = []
    if pin_left_black:
        # enforce/bias left-most pixel "black"
        seeds.append((0, black_label, pin_strength))

    H_l = build_Hl(n, seeds) if seeds else np.zeros_like(H_I)
    H = H_I + mu * H_l

    rho, Z, _ = gibbs_from_H(H, beta)

    # Only correlations with label site
    Sz_ops = [op_on(n, i, Sz) for i in range(n)]
    Sz_label = Sz_ops[label_site]

    SzSz = np.array([np.real(np.trace(rho @ (Sz_label @ Sz_ops[i])))
                     for i in range(n)])

    # match your earlier usage: Czz = 2 * <Sz_label Sz_i>
    Czz = 2.0 * SzSz
    return Czz, {"n": n, "edges": edges, "Z": Z, "H": H}



# Image-level: horizontal slices

def image_horizontal_slices_gibbs(img, beta, kappa=4.0,
                                  label_site=0,
                                  pin_left_black=True,
                                  pin_strength=20.0,
                                  mu=1.0,
                                  black_label=+1):
    """
    Apply exact 1D Gibbs row-by-row.

    Returns:
      C_map: shape (H, W) where row r contains 2< Sz_label Sz_i > for that row.
    """
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError("img must be a 2D array (H, W)")

    H, W = img.shape
    C_map = np.zeros((H, W), dtype=np.float64)

    row_meta = []
    for r in range(H):
        Czz, meta = row_SzSz_label_exact(
            img[r, :],
            beta=beta,
            kappa=kappa,
            label_site=label_site,
            pin_left_black=pin_left_black,
            pin_strength=pin_strength,
            mu=mu,
            black_label=black_label
        )
        C_map[r, :] = Czz
        row_meta.append(meta)

    return C_map, row_meta



# Segmentation helpers

def segment_from_correlations(C_map, threshold=0.0, black_value=0, white_value=255):
    """
    Convert correlation map C_map (H,W) into a segmentation mask.
    Rule: black if C >= threshold, else white.
    """
    seg = np.where(C_map >= threshold, black_value, white_value).astype(np.uint8)
    return seg

def enforce_left_black_per_row(seg, black_value=0, white_value=255):
    """
    Enforce that seg[r,0] is black_value for each row r by flipping that row if needed.
    """
    seg2 = seg.copy()
    H = seg2.shape[0]
    for r in range(H):
        if seg2[r, 0] != black_value:
            seg2[r, :] = np.where(seg2[r, :] == black_value, white_value, black_value)
    return seg2


def horizontal_slice_segmentation(img, beta, kappa=4.0,
                                  label_site=0,
                                  pin_strength=30.0,
                                  mu=1.0,
                                  black_label=+1,
                                  threshold=0.0,
                                  pin_mode="in_H"):
    """
    Full pipeline:
      - pin_mode="in_H": enforce left-most black by adding seed to Hamiltonian.
      - pin_mode="final_only": compute Gibbs WITHOUT pinning, then flip each row
        in the final segmentation so left-most pixel is black.

    Returns:
      seg (H,W) uint8 in {0,255},
      C_map (H,W) float,
      meta list
    """
    if pin_mode not in ("in_H", "final_only"):
        raise ValueError('pin_mode must be "in_H" or "final_only"')

    pin_in_H = (pin_mode == "in_H")

    C_map, meta = image_horizontal_slices_gibbs(
        img,
        beta=beta,
        kappa=kappa,
        label_site=label_site,
        pin_left_black=pin_in_H,
        pin_strength=pin_strength,
        mu=mu,
        black_label=black_label
    )

    seg = segment_from_correlations(C_map, threshold=threshold, black_value=0, white_value=255)

    if pin_mode == "final_only":
        seg = enforce_left_black_per_row(seg, black_value=0, white_value=255)

    return seg, C_map, meta



# -------------------------
# Example: toy 8x8 + MNIST 8x8
# -------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_openml
    from skimage.transform import resize


    # 1) TOY 8x8 EXAMPLE (guaranteed left-most pixel black)

    img8_toy = np.array([
        [0,   0,   0,   0, 255, 255, 255, 255],
        [0,   0,   0, 255, 255, 255, 255, 255],
        [0,   0, 255, 255, 255, 255, 255, 255],
        [0, 255, 255, 255, 255, 255, 255, 255],
        [0, 255, 255, 255, 255, 255, 255,   0],
        [0, 255, 255, 255, 255, 255,   0,   0],
        [0, 255, 255, 255, 255,   0,   0,   0],
        [0, 255, 255, 255,   0,   0,   0,   0],
    ], dtype=float)

    beta = 5.0
    kappa = 3.0
    pin_mode = "final_only"   # or "in_H"

    seg_toy, C_toy, _ = horizontal_slice_segmentation(
        img8_toy,
        beta=beta,
        kappa=kappa,
        label_site=0,
        pin_strength=30.0,
        mu=1.0,
        black_label=+1,
        threshold=0.0,
        pin_mode=pin_mode
    )


    # 2) MNIST → 8x8 EXAMPLE
  
    print("Loading MNIST...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    idx = 0   # change to explore
    img28 = X[idx].reshape(28, 28)
    label = y[idx]

    # Resize to 8x8
    img8_mnist = resize(
        img28,
        (8, 8),
        order=1,
        mode="reflect",
        anti_aliasing=True
    )

    # Rescale to [0,255]
    img8_mnist = img8_mnist / img8_mnist.max() * 255.0


    # Enforce left-most pixel black (boundary condition consistency)
    img8_mnist[:, 0] = 0.0

    # Optional: visualize row-wise sigmas to diagnose stability
    print("MNIST row sigmas:", [img8_mnist[r, :].std() for r in range(8)])

    seg_mnist, C_mnist, _ = horizontal_slice_segmentation(
        img8_mnist,
        beta=beta,
        kappa=kappa,
        label_site=0,
        pin_strength=30.0,
        mu=1.0,
        black_label=+1,
        threshold=0.0,
        pin_mode=pin_mode
    )


    # Plot results

    plt.figure(figsize=(14, 6))

    # ---- Toy ----
    plt.subplot(2, 3, 1)
    plt.imshow(img8_toy, cmap="gray", vmin=0, vmax=255)
    plt.title("Toy image (8×8)")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(C_toy, interpolation="nearest")
    plt.title(r"Toy: $2\langle S_0^z S_i^z\rangle$")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(seg_toy, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Toy segmentation\n(pin_mode={pin_mode})")
    plt.axis("off")

    # ---- MNIST ----
    plt.subplot(2, 3, 4)
    plt.imshow(img8_mnist, cmap="gray", vmin=0, vmax=255)
    plt.title(f"MNIST digit {label} → 8×8")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(C_mnist, interpolation="nearest")
    plt.title(r"MNIST: $2\langle S_0^z S_i^z\rangle$")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(seg_mnist, cmap="gray", vmin=0, vmax=255)
    plt.title(f"MNIST segmentation\n(pin_mode={pin_mode})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

