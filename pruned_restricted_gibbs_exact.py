import os
import glob
import zlib
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.sparse as sp
import scipy.linalg as la
from sklearn.datasets import fetch_openml
from skimage.transform import resize

# ============================================================
# Utilities
# ============================================================

def atomic_savez(path, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    np.savez(tmp, **kwargs)
    os.replace(tmp + ".npz", path)

def newest_ckpt(ckpt_dir):
    paths = glob.glob(os.path.join(ckpt_dir, "state_k_*.npz"))
    if not paths:
        return None, -1
    ks = []
    for p in paths:
        base = os.path.basename(p)
        k = int(base.split("_k_")[1].split(".npz")[0])
        ks.append((k, p))
    ks.sort()
    return ks[-1][1], ks[-1][0]

def state_signature(active_mask: np.ndarray, frozen_sz: np.ndarray) -> tuple[int, int]:
    a = np.asarray(active_mask, dtype=np.uint8)
    f = np.asarray(frozen_sz, dtype=np.float64)
    n_active = int(a.sum())
    h = zlib.crc32(a.tobytes())
    h = zlib.crc32(f.tobytes(), h)
    return n_active, int(h)

def sz_from_bit(bit):
    return 0.5 if bit else -0.5

def validate_anchor_spin(anchor_spin):
    if anchor_spin not in (-0.5, 0.5):
        raise ValueError("anchor_spin must be +0.5 or -0.5")

def validate_freeze_rule(rule):
    allowed = {
        "same_as_anchor",
        "opposite_of_anchor",
        "positive_to_down",
        "negative_to_down",
        "always_down",
        "always_up",
    }
    if rule not in allowed:
        raise ValueError(f"freeze_rule must be one of {sorted(allowed)}")

def correlation_from_fixed_anchor(mag_full, anchor_spin):
    """
    For a clamped anchor spin s_l in {+1/2, -1/2},
        2 <Sz_l Sz_i> = 2 * s_l * <Sz_i>.
    """
    validate_anchor_spin(anchor_spin)
    return 2.0 * float(anchor_spin) * np.asarray(mag_full, dtype=np.float64)

def assign_frozen_spins(corr_values, anchor_spin, rule="same_as_anchor"):
    """
    Assign frozen spins from correlation values according to a chosen policy.

    Parameters
    ----------
    corr_values : array-like
        Correlation values 2 <Sz_l Sz_i>.
    anchor_spin : float
        +0.5 or -0.5
    rule : str
        One of:
          - "same_as_anchor":
                corr >= 0 -> anchor_spin
                corr <  0 -> -anchor_spin

          - "opposite_of_anchor":
                corr >= 0 -> -anchor_spin
                corr <  0 -> anchor_spin

          - "positive_to_down":
                corr >= 0 -> -0.5
                corr <  0 -> +0.5

          - "negative_to_down":
                corr >= 0 -> +0.5
                corr <  0 -> -0.5

          - "always_down":
                everything -> -0.5

          - "always_up":
                everything -> +0.5
    """
    validate_anchor_spin(anchor_spin)
    validate_freeze_rule(rule)

    corr_values = np.asarray(corr_values, dtype=np.float64)

    if rule == "same_as_anchor":
        return np.where(corr_values >= 0.0, anchor_spin, -anchor_spin).astype(np.float64)

    if rule == "opposite_of_anchor":
        return np.where(corr_values >= 0.0, -anchor_spin, anchor_spin).astype(np.float64)

    if rule == "positive_to_down":
        return np.where(corr_values >= 0.0, -0.5, +0.5).astype(np.float64)

    if rule == "negative_to_down":
        return np.where(corr_values >= 0.0, +0.5, -0.5).astype(np.float64)

    if rule == "always_down":
        return np.full(corr_values.shape, -0.5, dtype=np.float64)

    if rule == "always_up":
        return np.full(corr_values.shape, +0.5, dtype=np.float64)

    raise RuntimeError("Unreachable freeze_rule branch")

# ============================================================
# Image + graph
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

def make_all_white_image(H=5, W=5, white=255.0):
    return np.full((H, W), white, dtype=float)

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

def neighbors_4(H, W):
    nbr = {}
    for r in range(H):
        for c in range(W):
            i = r * W + c
            out = []
            if r > 0:
                out.append((r - 1) * W + c)
            if r + 1 < H:
                out.append((r + 1) * W + c)
            if c > 0:
                out.append(r * W + (c - 1))
            if c + 1 < W:
                out.append(r * W + (c + 1))
            nbr[i] = out
    return nbr

def load_mnist_12x12(index=0):
    print("Loading MNIST (this may take a moment first time)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"]
    img28 = X[index].reshape(28, 28).astype(float)

    img12 = resize(img28, (12, 12), anti_aliasing=True, preserve_range=True)

    img12 = img12 - img12.min()
    if img12.max() > 0:
        img12 = 255.0 * img12 / img12.max()

    return img12.astype(float)

# ============================================================
# Basis + exact sector solvers
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

def build_xxz_sector_sparse_bitbasis(n, edges_active, local_fields, k):
    """
    Restricted XXZ Hamiltonian in k-up sector over ACTIVE spins only.

    Pair term:
        -Jz * Sz_i Sz_j
    XY flip term:
        -1/2 * (S_i^+ S_j^- + S_i^- S_j^+)
    Frozen-neighbor / fixed-anchor effect:
        -local_fields[i] * Sz_i
    """
    basis = basis_bitmasks_k(n, k)
    dim = len(basis)
    if dim == 0:
        return sp.csr_matrix((0, 0), dtype=np.float64), basis

    idx = {s: a for a, s in enumerate(basis)}

    diag = np.zeros(dim, dtype=np.float64)
    lf = np.asarray(local_fields, dtype=np.float64)
    if lf.shape[0] != n:
        raise ValueError("local_fields must have length n")

    for a, s in enumerate(basis):
        e = 0.0

        for (i, j, Jz) in edges_active:
            bi = (s >> i) & 1
            bj = (s >> j) & 1
            e += -Jz * sz_from_bit(bi) * sz_from_bit(bj)

        for i in range(n):
            hi = lf[i]
            if hi != 0.0:
                bi = (s >> i) & 1
                e += -hi * sz_from_bit(bi)

        diag[a] = e

    rows = list(range(dim))
    cols = list(range(dim))
    data = diag.tolist()

    if k > 0:
        for a, s in enumerate(basis):
            for (i, j, _) in edges_active:
                bi = (s >> i) & 1
                bj = (s >> j) & 1
                if bi != bj:
                    t = s ^ ((1 << i) | (1 << j))
                    b = idx.get(t, None)
                    if b is not None:
                        rows.append(a)
                        cols.append(b)
                        data.append(-0.5)

    Hk = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()
    return Hk, basis

def exact_magnetization_sector(Hk, basis, beta, n):
    """
    Exact computation in one sector:
      Zk = Tr(exp(-beta Hk))
      mag[i] = <Sz_i>
    """
    dim = len(basis)
    if dim == 0:
        return 0.0, np.zeros(n, dtype=np.float64)

    Hd = Hk.toarray()
    Hd = 0.5 * (Hd + Hd.T)

    evals, Q = la.eigh(Hd)
    w = np.exp(-beta * evals)
    diagW = (Q * Q) @ w
    Zk = float(diagW.sum())

    sz = np.empty((dim, n), dtype=np.float64)
    for a, s in enumerate(basis):
        for i in range(n):
            sz[a, i] = sz_from_bit((s >> i) & 1)

    mag = (sz.T @ diagW) / Zk
    return Zk, mag

def exact_observables_sector(Hk, basis, beta, n, label_site):
    """
    Exact computation in one sector:
      Zk = Tr(exp(-beta Hk))
      corr[i] = <Sz_label Sz_i>
      mag[i]  = <Sz_i>
    """
    dim = len(basis)
    if dim == 0:
        return 0.0, np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)

    Hd = Hk.toarray()
    Hd = 0.5 * (Hd + Hd.T)

    evals, Q = la.eigh(Hd)
    w = np.exp(-beta * evals)
    diagW = (Q * Q) @ w
    Zk = float(diagW.sum())

    sz = np.empty((dim, n), dtype=np.float64)
    for a, s in enumerate(basis):
        for i in range(n):
            sz[a, i] = sz_from_bit((s >> i) & 1)

    ls = int(label_site)
    if not (0 <= ls < n):
        raise ValueError("label_site out of range in reduced system")

    szl = sz[:, ls]
    O_corr = szl[:, None] * sz
    O_mag = sz

    corr = (O_corr.T @ diagW) / Zk
    mag = (O_mag.T @ diagW) / Zk
    return Zk, corr, mag

# ============================================================
# Pruning
# ============================================================

def is_locally_consistent(C, nbrs, i, tau_B=0.06, min_agree=3):
    neigh = nbrs[i]
    if len(neigh) == 0:
        return True
    diffs = [abs(C[i] - C[j]) for j in neigh]
    B = sum(diffs) / len(diffs)
    s_i = np.sign(C[i])
    agree = sum(1 for j in neigh if np.sign(C[j]) == s_i)
    return (B < tau_B) and (agree >= min(min_agree, len(neigh)))

def cap_freeze(freeze_mask, C_now, active_mask, cap_frac=1.0):
    if cap_frac >= 1.0:
        return freeze_mask
    cand = np.where(freeze_mask)[0]
    if cand.size == 0:
        return freeze_mask
    active_count = int(active_mask.sum())
    cap = max(1, int(np.ceil(cap_frac * active_count)))
    if cand.size <= cap:
        return freeze_mask
    scores = np.abs(C_now[cand])
    keep = cand[np.argsort(scores)[-cap:]]
    out = np.zeros_like(freeze_mask, dtype=bool)
    out[keep] = True
    return out

def compute_freeze_mask_window(
    C_hist, C_now, nbrs, active_mask,
    mode="buffer",
    eps=0.02, tau=0.20,
    tau_B=0.06, min_agree=3,
    cap_frac=1.0
):
    if mode not in ("buffer", "local", "stability_only"):
        raise ValueError('mode must be "buffer", "local", or "stability_only"')

    freeze = np.zeros_like(active_mask, dtype=bool)
    idx_act = np.where(active_mask)[0]
    if idx_act.size == 0:
        return freeze

    diffmax = np.zeros(idx_act.size, dtype=np.float64)
    for Cold in C_hist:
        diffmax = np.maximum(diffmax, np.abs(C_now[idx_act] - Cold[idx_act]))

    stable = np.zeros_like(active_mask, dtype=bool)
    stable[idx_act] = (diffmax <= eps)
    stable &= (np.abs(C_now) >= tau)

    if mode == "stability_only":
        freeze = stable
    elif mode == "local":
        for i in np.where(stable)[0]:
            if is_locally_consistent(C_now, nbrs, i, tau_B=tau_B, min_agree=min_agree):
                freeze[i] = True
    else:
        unstable = np.zeros_like(active_mask, dtype=bool)
        unstable[idx_act] = (diffmax > eps)

        neighbor_of_unstable = np.zeros_like(active_mask, dtype=bool)
        for j in np.where(unstable)[0]:
            for i in nbrs[j]:
                if active_mask[i]:
                    neighbor_of_unstable[i] = True

        freeze = stable & (~neighbor_of_unstable)

    freeze = cap_freeze(freeze, C_now, active_mask, cap_frac=cap_frac)
    return freeze

def remap_edges_with_fields(edges0, active_mask, frozen_sz, n0):
    """
    Build reduced active-active edge list and local fields induced by frozen spins.
    """
    active_sites = [i for i in range(n0) if active_mask[i]]
    old_to_new = {old: new for new, old in enumerate(active_sites)}

    new_edges = []
    local_fields = np.zeros(len(active_sites), dtype=np.float64)

    for (i, j, Jz) in edges0:
        i_active = active_mask[i]
        j_active = active_mask[j]

        if i_active and j_active:
            new_edges.append((old_to_new[i], old_to_new[j], Jz))
        elif i_active and (not j_active):
            s_j = float(frozen_sz[j])
            local_fields[old_to_new[i]] += Jz * s_j
        elif (not i_active) and j_active:
            s_i = float(frozen_sz[i])
            local_fields[old_to_new[j]] += Jz * s_i

    return new_edges, active_sites, old_to_new, local_fields

# ============================================================
# Plotting
# ============================================================

def save_correlation_plot(
    img, C_map, beta, kappa, k, outpath,
    title_prefix="Upto",
    active_mask=None,
    frozen_sz=None,
    anchor_site=None,
    anchor_spin=None,
    freeze_label=False
):
    H, W = img.shape
    plt.figure(figsize=(10, 4))

    ax0 = plt.subplot(1, 2, 1)
    ax0.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")

    if active_mask is not None:
        frozen = ~active_mask.reshape(H, W)

        if frozen_sz is None:
            rr, cc = np.where(frozen)
            ax0.scatter(cc, rr, s=60, facecolors="none", edgecolors="cyan", linewidths=1.5)
        else:
            fsz = frozen_sz.reshape(H, W)
            up = frozen & (fsz > 0)
            dn = frozen & (fsz < 0)

            r1, c1 = np.where(up)
            r2, c2 = np.where(dn)

            if len(r1) > 0:
                ax0.scatter(c1, r1, s=60, facecolors="none", edgecolors="lime", linewidths=1.8, label="frozen +")
            if len(r2) > 0:
                ax0.scatter(c2, r2, s=60, facecolors="none", edgecolors="red", linewidths=1.8, label="frozen -")
            if (len(r1) + len(r2)) > 0:
                ax0.legend(loc="lower right", fontsize=8, framealpha=0.7)

        ax0.set_title(f"Image (frozen={int(frozen.sum())}/{H*W})")
    else:
        ax0.set_title("Image")

    if anchor_site is not None:
        rr = anchor_site // W
        cc = anchor_site % W
        marker_label = "anchor frozen" if freeze_label else "anchor active"
        ax0.scatter([cc], [rr], s=120, marker="*", c="yellow",
                    edgecolors="black", linewidths=0.8, label=marker_label)

    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = plt.subplot(1, 2, 2)
    im = ax1.imshow(C_map, interpolation="nearest")
    if freeze_label:
        mode_text = f"anchor fixed {float(anchor_spin):+.1f}"
    else:
        mode_text = "anchor active"
    ax1.set_title(
        f"{title_prefix} k={k}: 2<Sz_l Sz_i>\n"
        f"({mode_text}, beta={beta}, kappa={kappa})"
    )
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()

def classify_from_C(C_map, tau_class=0.0, pos_value=255, neg_value=0, unk_value=127):
    C = np.asarray(C_map)
    out = np.full(C.shape, unk_value, dtype=np.uint8)

    if tau_class <= 0.0:
        out = np.where(C >= 0.0, pos_value, neg_value).astype(np.uint8)
        return out

    out[C >= +tau_class] = pos_value
    out[C <= -tau_class] = neg_value
    return out

def save_classification_plot(
    img, cls_map, outpath, title="FINAL classification",
    active_mask=None, anchor_site=None
):
    H, W = img.shape
    plt.figure(figsize=(10, 4))

    ax0 = plt.subplot(1, 2, 1)
    ax0.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")

    if active_mask is not None:
        frozen = ~active_mask.reshape(H, W)
        rr, cc = np.where(frozen)
        ax0.scatter(cc, rr, s=60, facecolors="none", edgecolors="cyan", linewidths=1.5)
        ax0.set_title(f"Image (frozen={int(frozen.sum())}/{H*W})")
    else:
        ax0.set_title("Image")

    if anchor_site is not None:
        rr = anchor_site // W
        cc = anchor_site % W
        ax0.scatter([cc], [rr], s=120, marker="*", c="yellow",
                    edgecolors="black", linewidths=0.8)

    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = plt.subplot(1, 2, 2)
    ax1.imshow(cls_map, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax1.set_title(title)
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()

# ============================================================
# Main exact resumable runner
# ============================================================

def run_resumable_exact(
    img, beta, kappa, k_top,
    label_site=0,
    anchor_spin=-0.5,
    freeze_label=False,
    freeze_rule="same_as_anchor",
    k_freeze_start=3,
    eps=0.02, tau=0.20, tau_B=0.06, min_agree=3,
    prune_mode="buffer",
    cap_frac=1.0,
    window=3,
    out_dir="out_exact_pruned",
    exact_max_dim=4000,
    do_pruning=True
):
    """
    Exact pruned restricted Gibbs in two modes:

    1) freeze_label = False
       - label remains active
       - exact correlations are computed directly:
             C_i = 2 <Sz_l Sz_i>

    2) freeze_label = True
       - label is clamped to anchor_spin in {+1/2, -1/2}
       - exact magnetizations are computed
       - correlations are reconstructed by
             C_i = 2 * anchor_spin * <Sz_i>

    In both modes:
       - pruning is done on correlation
       - frozen spins induce local fields on active neighbors
       - frozen spin values are assigned by `freeze_rule`
    """
    validate_anchor_spin(anchor_spin)
    validate_freeze_rule(freeze_rule)

    if window < 2:
        raise ValueError("window must be >= 2")

    img = np.asarray(img, dtype=float)
    H, W = img.shape
    n0 = H * W
    if not (0 <= label_site < n0):
        raise ValueError("label_site out of range for original lattice")

    _, _, _, edges0 = build_Jz_edges_2d(img, kappa=kappa)
    nbrs0 = neighbors_4(H, W)

    plots_dir = os.path.join(out_dir, "plots")
    sectors_dir = os.path.join(out_dir, "sectors")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(sectors_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path, last_k = newest_ckpt(ckpt_dir)
    if ckpt_path is None:
        print("No checkpoint found. Starting fresh.")
        active_mask = np.ones(n0, dtype=bool)
        frozen_sz = np.zeros(n0, dtype=np.float64)

        if freeze_label:
            active_mask[label_site] = False
            frozen_sz[label_site] = float(anchor_spin)
        else:
            active_mask[label_site] = True
            frozen_sz[label_site] = 0.0

        Z_cum = 0.0
        Num_cum = np.zeros(n0, dtype=np.float64)
        Mag_cum = np.zeros(n0, dtype=np.float64)
        C_hist = [np.zeros(n0, dtype=np.float64) for _ in range(window - 1)]
        start_k = 0
    else:
        print(f"Resuming from: {ckpt_path}")
        d = np.load(ckpt_path, allow_pickle=False)
        active_mask = d["active_mask"].astype(np.uint8).astype(bool)
        frozen_sz = d["frozen_sz"].astype(np.float64)
        Z_cum = float(d["Z_cum"])
        Num_cum = d["Num_cum"].astype(np.float64)
        Mag_cum = d["Mag_cum"].astype(np.float64)

        if freeze_label:
            active_mask[label_site] = False
            frozen_sz[label_site] = float(anchor_spin)
        else:
            active_mask[label_site] = True
            if label_site < frozen_sz.shape[0]:
                frozen_sz[label_site] = 0.0

        if "C_hist" in d:
            C_hist_stack = d["C_hist"].astype(np.float64)
            C_hist = [C_hist_stack[t].copy() for t in range(C_hist_stack.shape[0])]
        else:
            C_hist = [np.zeros(n0, dtype=np.float64) for _ in range(window - 1)]

        if len(C_hist) != window - 1:
            if len(C_hist) > window - 1:
                C_hist = C_hist[:window - 1]
            else:
                while len(C_hist) < window - 1:
                    C_hist.append(C_hist[-1].copy())

        start_k = last_k + 1
        print(f"Last completed k={last_k}; continuing at k={start_k}")

    for k in range(start_k, k_top + 1):
        edges_active, active_sites, old_to_new, local_fields = remap_edges_with_fields(
            edges0, active_mask, frozen_sz, n0
        )
        current_n = len(active_sites)

        print(f"\n=== k = {k} ===")
        print(f"active spins = {current_n}")

        n_active, sig = state_signature(active_mask, frozen_sz)
        sec_path = os.path.join(sectors_dir, f"sector_k_{k}.npz")

        loaded_ok = False
        if os.path.exists(sec_path):
            dd = np.load(sec_path, allow_pickle=False)
            if ("sig" in dd) and ("n_active" in dd):
                if int(dd["n_active"]) == n_active and int(dd["sig"]) == sig:
                    Zk = float(dd["Zk"])
                    corr_k = dd["corr_k"].astype(np.float64)
                    mag_k = dd["mag_k"].astype(np.float64)
                    loaded_ok = True
                    print(
                        f"[sector k={k}] loaded checkpoint "
                        f"(len(corr_k)={corr_k.size}, len(mag_k)={mag_k.size})"
                    )
                else:
                    print(f"[sector k={k}] checkpoint mismatch; recomputing")
            else:
                print(f"[sector k={k}] old-format checkpoint; recomputing")

        if not loaded_ok:
            if k > current_n:
                print(f"[sector k={k}] skip (k > active spins)")
                Zk = 0.0
                corr_k = np.zeros(current_n, dtype=np.float64)
                mag_k = np.zeros(current_n, dtype=np.float64)
            else:
                Hk, basis = build_xxz_sector_sparse_bitbasis(
                    current_n, edges_active, local_fields, k
                )
                dimk = Hk.shape[0]
                print(f"[sector k={k}] dim(Hk) = {dimk}")

                if dimk == 0:
                    Zk = 0.0
                    corr_k = np.zeros(current_n, dtype=np.float64)
                    mag_k = np.zeros(current_n, dtype=np.float64)

                elif dimk == 1:
                    h = float(Hk[0, 0])
                    Zk = float(np.exp(-beta * h))
                    s = basis[0]
                    mag_k = np.array(
                        [sz_from_bit((s >> i) & 1) for i in range(current_n)],
                        dtype=np.float64
                    )

                    if freeze_label:
                        corr_k = correlation_from_fixed_anchor(mag_k, anchor_spin)
                    else:
                        if not active_mask[label_site]:
                            raise RuntimeError("label_site unexpectedly inactive in active-label mode.")
                        label_new = old_to_new[label_site]
                        corr_k = mag_k[label_new] * mag_k

                else:
                    if dimk > exact_max_dim:
                        raise RuntimeError(
                            f"dim(Hk)={dimk} exceeds exact_max_dim={exact_max_dim}. "
                            "Lower k_top, prune more, reduce image size, or increase exact_max_dim."
                        )

                    if freeze_label:
                        Zk, mag_k = exact_magnetization_sector(Hk, basis, beta=beta, n=current_n)
                        corr_k = correlation_from_fixed_anchor(mag_k, anchor_spin)
                    else:
                        if not active_mask[label_site]:
                            raise RuntimeError("label_site unexpectedly inactive in active-label mode.")
                        label_new = old_to_new[label_site]
                        Zk, corr_k, mag_k = exact_observables_sector(
                            Hk, basis, beta=beta, n=current_n, label_site=label_new
                        )

            atomic_savez(
                sec_path,
                Zk=np.array(Zk, dtype=np.float64),
                corr_k=np.asarray(corr_k, dtype=np.float64),
                mag_k=np.asarray(mag_k, dtype=np.float64),
                n_active=np.array(n_active, dtype=np.int64),
                sig=np.array(sig, dtype=np.int64),
            )
            print(f"[sector k={k}] saved checkpoint -> {sec_path}")

        # Lift sector observables to full lattice
        corr_full = np.zeros(n0, dtype=np.float64)
        mag_full = np.zeros(n0, dtype=np.float64)

        corr_full[active_mask] = corr_k
        mag_full[active_mask] = mag_k

        if freeze_label:
            mag_full[~active_mask] = frozen_sz[~active_mask]
            corr_full[~active_mask] = correlation_from_fixed_anchor(
                mag_full[~active_mask], anchor_spin
            )
        else:
            # label stays active; frozen non-label sites are approximated via
            # <Sz_l Sz_i> = frozen_sz[i] * <Sz_l>
            if not active_mask[label_site]:
                raise RuntimeError("label_site unexpectedly inactive in active-label mode.")
            label_new = old_to_new[label_site]
            m_label = float(mag_k[label_new])

            frozen_nonlabel = (~active_mask).copy()
            frozen_nonlabel[label_site] = False

            mag_full[frozen_nonlabel] = frozen_sz[frozen_nonlabel]
            corr_full[frozen_nonlabel] = frozen_sz[frozen_nonlabel] * m_label

            # active label site already set through active branch
            mag_full[label_site] = mag_k[label_new]
            corr_full[label_site] = corr_k[label_new]

        # Accumulate up to k
        Z_cum += Zk
        Num_cum += Zk * corr_full
        Mag_cum += Zk * mag_full

        corr_upto = Num_cum / Z_cum
        mag_upto = Mag_cum / Z_cum
        C_now = 2.0 * corr_upto
        C_map = C_now.reshape(H, W)

        plot_path = os.path.join(plots_dir, f"corr_upto_{k}.png")
        if not os.path.exists(plot_path):
            save_correlation_plot(
                img, C_map, beta, kappa, k,
                plot_path,
                title_prefix=f"EXACT-{prune_mode}-win{window}",
                active_mask=active_mask,
                frozen_sz=frozen_sz,
                anchor_site=label_site,
                anchor_spin=anchor_spin,
                freeze_label=freeze_label,
            )
            print(f"[upto k={k}] saved plot -> {plot_path}")
        else:
            print(f"[upto k={k}] plot exists; skipping")

        # Pruning decision on correlation
        if do_pruning and (k >= k_freeze_start) and (current_n > 0) and ((window - 1) > 0):
            freeze = compute_freeze_mask_window(
                C_hist=C_hist,
                C_now=C_now,
                nbrs=nbrs0,
                active_mask=active_mask,
                mode=prune_mode,
                eps=eps, tau=tau,
                tau_B=tau_B, min_agree=min_agree,
                cap_frac=cap_frac
            )

            if not freeze_label:
                freeze[label_site] = False

            nf = int(freeze.sum())
        else:
            freeze = np.zeros(n0, dtype=bool)
            if not freeze_label:
                freeze[label_site] = False
            nf = 0

        print(f"[prune] freeze candidates = {nf}")

        if nf > 0:
            idxs = np.where(freeze)[0]
            frozen_sz[idxs] = assign_frozen_spins(
                C_now[idxs],
                anchor_spin=anchor_spin,
                rule=freeze_rule
            )
            active_mask[idxs] = False
            print(f"[prune] new active spins = {int(active_mask.sum())}")

        if len(C_hist) > 0:
            C_hist = [C_now] + C_hist[:-1]

        ckpt_out = os.path.join(ckpt_dir, f"state_k_{k}.npz")
        atomic_savez(
            ckpt_out,
            active_mask=active_mask.astype(np.uint8),
            frozen_sz=frozen_sz.astype(np.float64),
            Z_cum=np.array(Z_cum, dtype=np.float64),
            Num_cum=Num_cum.astype(np.float64),
            Mag_cum=Mag_cum.astype(np.float64),
            C_hist=np.stack(C_hist, axis=0).astype(np.float64)
                if len(C_hist) > 0 else np.zeros((0, n0), dtype=np.float64),
            window=np.array(window, dtype=np.int64),
        )
        print(f"[checkpoint] saved -> {ckpt_out}")

    # Final correlation map
    corr_final = Num_cum / Z_cum
    C_final = 2.0 * corr_final
    C_final_map = C_final.reshape(H, W)

    final_path = os.path.join(out_dir, "final_correlation_map.png")
    save_correlation_plot(
        img, C_final_map, beta, kappa, k="final",
        outpath=final_path,
        title_prefix=f"FINAL EXACT {prune_mode}-win{window}",
        active_mask=active_mask,
        frozen_sz=frozen_sz,
        anchor_site=label_site,
        anchor_spin=anchor_spin,
        freeze_label=freeze_label,
    )
    print(f"[final] saved -> {final_path}")

    TAU_CLASS = 0.0
    cls_final = classify_from_C(
        C_final_map,
        tau_class=TAU_CLASS,
        pos_value=255,
        neg_value=0,
        unk_value=127
    )

    cls_path = os.path.join(out_dir, "final_classification_map.png")
    save_classification_plot(
        img, cls_final,
        outpath=cls_path,
        title=f"FINAL classification from 2<Sz_l Sz_i> (tau_class={TAU_CLASS})",
        active_mask=active_mask,
        anchor_site=label_site,
    )
    print(f"[final] classification saved -> {cls_path}")

    print("\nDone. Output in:", out_dir)

# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    H = 5
    W = 5

    # Choose one:
    img = make_plus_image(H=H, W=W, white=255.0, black=0.0, thickness=1)
    # img = make_all_white_image(H=H, W=W, white=255.0)
    # img = load_mnist_12x12(index=0)

    beta = 10.0
    kappa = 3.0
    k_top = 9

    WINDOW = 3
    PRUNE_MODE = "stability_only"
    CAP_FRAC = 1.0
    EPS = 0.007
    TAU = 0.005

    ANCHOR_SPIN = -0.5
    FREEZE_LABEL = False          # False: label stays active; True: label fixed from start
    FREEZE_RULE = "same_as_anchor"  # try also: "positive_to_down", "negative_to_down"

    out_dir = (
        f"out_exact_"
        f"{H}x{W}_"
        f"{PRUNE_MODE}_"
        f"win{WINDOW}_"
        f"eps{EPS}_"
        f"tau{TAU}_"
        f"anchor{ANCHOR_SPIN:+.1f}_"
        f"freezeLabel{int(FREEZE_LABEL)}_"
        f"rule_{FREEZE_RULE}"
    )

    run_resumable_exact(
        img,
        beta=beta,
        kappa=kappa,
        k_top=k_top,
        label_site=0,
        anchor_spin=ANCHOR_SPIN,
        freeze_label=FREEZE_LABEL,
        freeze_rule=FREEZE_RULE,
        k_freeze_start=3,
        eps=EPS,
        tau=TAU,
        tau_B=0.06,
        min_agree=3,
        prune_mode=PRUNE_MODE,
        cap_frac=CAP_FRAC,
        window=WINDOW,
        out_dir=out_dir,
        exact_max_dim=20000,
        do_pruning=True
    )