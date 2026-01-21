import numpy as np
import matplotlib.pyplot as plt
import time


def build_Jz_edges_2d(img, kappa=4.0):
    print("[1] Building 2D Jz edges...")
    img = np.asarray(img, dtype=float)
    H, W = img.shape

    sigma = img.std(ddof=0)
    sigma = sigma if sigma > 0 else 1.0
    print(f"    Image size = {H}x{W}, sigma = {sigma:.4f}")

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

    print(f"    Number of edges = {len(edges)}")
    return H * W, H, W, edges


def sz_from_bit(bit):
    return 0.5 if bit else -0.5


def build_xxz_sector_H(n, edges, k_up=2):
    print("[2] Building XXZ Hamiltonian in fixed sector...")
    t0 = time.time()

    basis = [s for s in range(1 << n) if s.bit_count() == k_up]
    dim = len(basis)
    index = {s: a for a, s in enumerate(basis)}

    print(f"    n = {n}, N_up = {k_up}")
    print(f"    Sector dimension = {dim}")

    Hk = np.zeros((dim, dim), dtype=np.float64)

    for a, s in enumerate(basis):
        if a % max(1, dim // 10) == 0:
            print(f"    Filling row {a+1}/{dim}")

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
                Hk[a, index[t]] += -0.5

    print(f"    Hamiltonian build time: {time.time() - t0:.2f}s")
    return Hk, basis


def gibbs_from_H(H, beta):
    print("[3] Diagonalizing Hamiltonian...")
    t0 = time.time()

    evals, U = np.linalg.eigh(H)
    w = np.exp(-beta * evals)
    Z = float(w.sum())
    rho = (U * (w / Z)) @ U.T

    print(f"    Diagonalization time: {time.time() - t0:.2f}s")
    print(f"    Partition function Z = {Z:.6e}")
    return rho, Z, evals


def correlations_SzSz_label_in_sector(n, basis, rho, label_site=0):
    print("[4] Computing correlations...")
    p = np.real(np.diag(rho))
    corr = np.zeros(n, dtype=np.float64)

    for a, s in enumerate(basis):
        if a % max(1, len(basis) // 10) == 0:
            print(f"    Correlation accumulation {a+1}/{len(basis)}")

        pa = p[a]
        szl = sz_from_bit((s >> label_site) & 1)
        for i in range(n):
            szi = sz_from_bit((s >> i) & 1)
            corr[i] += pa * (szl * szi)

    return corr


def segment_from_correlations(C_map, threshold=0.0):
    print("[5] Thresholding correlations → segmentation")
    return np.where(C_map >= threshold, 0, 255).astype(np.uint8)


def enforce_label_black_global(seg, label_site, H, W):
    r0 = label_site // W
    c0 = label_site % W
    if seg[r0, c0] != 0:
        print("    Flipping segmentation to enforce label-site black")
        seg = np.where(seg == 0, 255, 0).astype(seg.dtype)
    return seg


def whole_image_xxz_sector_segmentation(
    img,
    beta,
    kappa=4.0,
    k_up=2,
    label_site=0,
    threshold=0.0,
    enforce_label_black_final=True,
):
    print("========== XXZ SECTOR SEGMENTATION ==========")
    print("[0] Starting segmentation pipeline")

    n, H, W, edges = build_Jz_edges_2d(img, kappa=kappa)
    Hk, basis = build_xxz_sector_H(n, edges, k_up=k_up)
    rho, Z, _ = gibbs_from_H(Hk, beta)

    corr = correlations_SzSz_label_in_sector(n, basis, rho, label_site=label_site)
    Czz = 2.0 * corr
    C_map = Czz.reshape(H, W)

    seg = segment_from_correlations(C_map, threshold=threshold)

    if enforce_label_black_final:
        seg = enforce_label_black_global(seg, label_site, H, W)

    print("[6] Segmentation complete")
    print("============================================")

    return seg, C_map

# -------------------------
# Example: toy 8x8 + MNIST/digits 8x8
# -------------------------
if __name__ == "__main__":
    beta = 2.0
    kappa = 3.0
    k_up = 2
    label_site = 0
    threshold = 0.0

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

    seg_toy, C_toy, meta_toy = whole_image_xxz_sector_segmentation(
        img8_toy,
        beta=beta,
        kappa=kappa,
        k_up=k_up,
        label_site=label_site,
        threshold=threshold,
        enforce_label_black_final=True,
    )

    img8_data = None
    data_title = None

    try:
        from sklearn.datasets import fetch_openml
        from skimage.transform import resize

        print("Loading MNIST from OpenML...")
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        idx = 0
        img28 = X[idx].reshape(28, 28)
        label = y[idx]

        img8 = resize(img28, (8, 8), order=1, mode="reflect", anti_aliasing=True)
        img8 = img8 / (img8.max() if img8.max() > 0 else 1.0) * 255.0

        img8_data = img8
        data_title = f"MNIST {label} → 8×8"

    except Exception as e:
        print("OpenML MNIST failed (using sklearn digits 8x8 instead).")
        print("Reason:", repr(e))
        from sklearn.datasets import load_digits

        digits = load_digits()
        idx = 0
        img8 = digits.images[idx].astype(float)
        label = digits.target[idx]
        img8 = img8 / (img8.max() if img8.max() > 0 else 1.0) * 255.0

        img8_data = img8
        data_title = f"Digits {label} (8×8)"

    seg_data, C_data, meta_data = whole_image_xxz_sector_segmentation(
        img8_data,
        beta=beta,
        kappa=kappa,
        k_up=k_up,
        label_site=label_site,
        threshold=threshold,
        enforce_label_black_final=True,
    )

    plt.figure(figsize=(14, 6))

    plt.subplot(2, 3, 1)
    plt.imshow(img8_toy, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    plt.title("Toy image (8×8)")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(C_toy, interpolation="nearest")
    plt.title(r"Toy: $2\langle S_\ell^z S_i^z\rangle$")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(seg_toy, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    plt.title(f"Toy seg (XXZ, N_up={k_up})")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(img8_data, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    plt.title(data_title)
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(C_data, interpolation="nearest")
    plt.title(r"Data: $2\langle S_\ell^z S_i^z\rangle$")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(seg_data, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    plt.title(f"Data seg (XXZ, N_up={k_up})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
