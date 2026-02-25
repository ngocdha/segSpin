import os
import glob
import zlib
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.datasets import fetch_openml
from skimage.transform import resize

# -------------------------
# Utilities
# -------------------------

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


def active_signature(active_mask: np.ndarray) -> tuple[int, int]:
    a = np.asarray(active_mask, dtype=np.uint8)
    return int(a.sum()), int(zlib.crc32(a.tobytes()))


def sz_from_bit(bit):
    return 0.5 if bit else -0.5


def pack_edges(edges):
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float64)
    ij = np.array([(i, j) for (i, j, _) in edges], dtype=np.int32)
    Jz = np.array([J for (_, _, J) in edges], dtype=np.float64)
    return ij, Jz


def unpack_edges(edges_ij, edges_Jz):
    m = edges_Jz.shape[0]
    return [(int(edges_ij[t, 0]), int(edges_ij[t, 1]), float(edges_Jz[t])) for t in range(m)]


# -------------------------
# Image + graph
# -------------------------

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


# -------------------------
# Basis + XXZ sector
# -------------------------

def basis_bitmasks_k(n, k):
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
    basis = basis_bitmasks_k(n, k)
    dim = len(basis)
    if dim == 0:
        return sp.csr_matrix((0, 0), dtype=np.float64), basis

    idx = {s: a for a, s in enumerate(basis)}

    diag = np.zeros(dim, dtype=np.float64)
    for a, s in enumerate(basis):
        e = 0.0
        for (i, j, Jz) in edges:
            bi = (s >> i) & 1
            bj = (s >> j) & 1
            e += -Jz * sz_from_bit(bi) * sz_from_bit(bj)
        diag[a] = e

    rows = list(range(dim))
    cols = list(range(dim))
    data = diag.tolist()

    if k > 0:
        for a, s in enumerate(basis):
            for (i, j, _) in edges:
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


def hutchinson_szsz_sector(Hk, basis, beta, n, label_site=0, n_samples=64, seed=0):
    rng = np.random.default_rng(seed)
    dim = len(basis)

    sz = np.empty((dim, n), dtype=np.float64)
    for a, s in enumerate(basis):
        for i in range(n):
            sz[a, i] = sz_from_bit((s >> i) & 1)

    ls = label_site
    szl = sz[:, ls]
    O_diag = szl[:, None] * sz

    Z_acc = 0.0
    N_acc = np.zeros(n, dtype=np.float64)

    for _ in range(n_samples):
        v = rng.integers(0, 2, size=dim) * 2 - 1
        v = v.astype(np.float64)
        w = spla.expm_multiply((-beta) * Hk, v)
        Z_acc += float(np.dot(v, w))
        tmp = v * w
        N_acc += (O_diag.T @ tmp)

    Zk = Z_acc / n_samples
    Nk = N_acc / n_samples
    corr_k = Nk / Zk
    return Zk, corr_k


# -------------------------
# Pruning (generic window length)
# -------------------------

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
    else:  # buffer
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


def remap_edges(edges0, active_mask, n0):
    active_sites = [i for i in range(n0) if active_mask[i]]
    old_to_new = {old: new for new, old in enumerate(active_sites)}
    new_edges = []
    for (i, j, Jz) in edges0:
        if active_mask[i] and active_mask[j]:
            new_edges.append((old_to_new[i], old_to_new[j], Jz))
    return new_edges, active_sites



# -------------------------
# Plotting
# -------------------------

def save_cmap_plot(img, C_map, beta, kappa, k, outpath,
                   title_prefix="Upto",
                   active_mask=None,
                   frozen_sz=None):
    H, W = img.shape

    plt.figure(figsize=(10, 4))

    ax0 = plt.subplot(1, 2, 1)
    ax0.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")

    if active_mask is not None:
        frozen = ~active_mask.reshape(H, W)
        rr, cc = np.where(frozen)

        if frozen_sz is None:
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

        n_frozen = int(frozen.sum())
        ax0.set_title(f"Image (frozen={n_frozen}/{H*W})")
    else:
        ax0.set_title("Image")

    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = plt.subplot(1, 2, 2)
    im = ax1.imshow(C_map, interpolation="nearest")
    ax1.set_title(f"{title_prefix} k={k}: 2<Sz_l Sz_i>\n(beta={beta}, kappa={kappa})")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()

def classify_from_C(C_map, tau_class=0.0, pos_value=255, neg_value=0, unk_value=127):
    """
    Classify pixels from correlation map C_map (= 2< Sz_l Sz_i >).

    If tau_class == 0:
      C >= 0 -> pos_value, else neg_value

    If tau_class > 0:
      C >= +tau_class -> pos_value
      C <= -tau_class -> neg_value
      otherwise -> unk_value
    """
    C = np.asarray(C_map)
    out = np.full(C.shape, unk_value, dtype=np.uint8)

    if tau_class <= 0.0:
        out = np.where(C >= 0.0, pos_value, neg_value).astype(np.uint8)
        return out

    out[C >= +tau_class] = pos_value
    out[C <= -tau_class] = neg_value
    return out


def save_classification_plot(img, cls_map, outpath, title="FINAL classification", active_mask=None):
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


# -------------------------
# Main resumable runner
# -------------------------

def load_mnist_12x12(index=0):
    """
    Load one MNIST digit, resize to 12x12, return float image in [0,255].
    """
    print("Loading MNIST (this may take a moment first time)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"]
    y = mnist["target"]

    img28 = X[index].reshape(28, 28).astype(float)

    # resize to 12x12
    img12 = resize(img28, (12, 12), anti_aliasing=True, preserve_range=True)

    # normalize to [0,255]
    img12 = img12 - img12.min()
    if img12.max() > 0:
        img12 = 255.0 * img12 / img12.max()

    return img12.astype(float)

def run_resumable(
    img, beta, kappa, k_top,
    label_site=0, n_samples=64, seed=0,
    k_freeze_start=3,
    eps=0.02, tau=0.20, tau_B=0.06, min_agree=3,
    prune_mode="buffer",
    cap_frac=1.0,
    window=3,
    out_dir="out_pruned_resumable"
):
    if window < 2:
        raise ValueError("window must be >= 2")

    img = np.asarray(img, dtype=float)
    H, W = img.shape
    n0 = H * W

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
        edges = edges0
        current_n = n0
        Z_cum = 0.0
        Num_cum = np.zeros(n0, dtype=np.float64)
        C_hist = [np.zeros(n0, dtype=np.float64) for _ in range(window - 1)]
        start_k = 0
    else:
        print(f"Resuming from: {ckpt_path}")
        d = np.load(ckpt_path, allow_pickle=False)
        active_mask = d["active_mask"].astype(bool)
        frozen_sz = d["frozen_sz"].astype(np.float64)

        edges = unpack_edges(d["edges_ij"].astype(np.int32), d["edges_Jz"].astype(np.float64))
        current_n = int(d["current_n"])
        Z_cum = float(d["Z_cum"])
        Num_cum = d["Num_cum"].astype(np.float64)

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
        print(f"\n=== k = {k} ===")
        print(f"active spins = {current_n}")

        n_active, sig = active_signature(active_mask)
        sec_path = os.path.join(sectors_dir, f"sector_k_{k}.npz")

        loaded_ok = False
        if os.path.exists(sec_path):
            dd = np.load(sec_path, allow_pickle=False)
            if ("sig" in dd) and ("n_active" in dd):
                if int(dd["n_active"]) == n_active and int(dd["sig"]) == sig:
                    Zk = float(dd["Zk"])
                    corr_k = dd["corr_k"].astype(np.float64)
                    loaded_ok = True
                    print(f"[sector k={k}] loaded checkpoint (len(corr_k)={corr_k.size})")
                else:
                    print(f"[sector k={k}] checkpoint mismatch; recomputing")
            else:
                print(f"[sector k={k}] old-format checkpoint; recomputing")

        if not loaded_ok:
            if k > current_n:
                print(f"[sector k={k}] skip (k > active spins)")
                Zk = 0.0
                corr_k = np.zeros(current_n, dtype=np.float64)
            else:
                Hk, basis = build_xxz_sector_sparse_bitbasis(current_n, edges, k)
                dimk = Hk.shape[0]
                print(f"[sector k={k}] dim(Hk) = {dimk}")

                if dimk == 0:
                    Zk = 0.0
                    corr_k = np.zeros(current_n, dtype=np.float64)
                elif dimk == 1:
                    h = float(Hk[0, 0])
                    Zk = float(np.exp(-beta * h))
                    s = basis[0]
                    ls = min(label_site, current_n - 1)
                    szl = sz_from_bit((s >> ls) & 1)
                    corr_k = np.array([szl * sz_from_bit((s >> i) & 1) for i in range(current_n)], dtype=np.float64)
                else:
                    Zk, corr_k = hutchinson_szsz_sector(
                        Hk, basis, beta=beta, n=current_n,
                        label_site=min(label_site, current_n - 1),
                        n_samples=n_samples,
                        seed=seed + 101 * k
                    )

            atomic_savez(
                sec_path,
                Zk=np.array(Zk, dtype=np.float64),
                corr_k=np.asarray(corr_k, dtype=np.float64),
                n_active=np.array(n_active, dtype=np.int64),
                sig=np.array(sig, dtype=np.int64),
            )
            print(f"[sector k={k}] saved checkpoint -> {sec_path}")

        # lift to full
        corr_full = np.zeros(n0, dtype=np.float64)
        corr_full[active_mask] = corr_k

        # frozen approx
        if not active_mask[label_site]:
            sz_l = frozen_sz[label_site]
        else:
            sz_l = 0.5 if C_hist[0][label_site] >= 0 else -0.5
        corr_full[~active_mask] = frozen_sz[~active_mask] * sz_l

        # accumulate upto
        Z_cum += Zk
        Num_cum += Zk * corr_full

        corr_upto = Num_cum / Z_cum
        C_now = 2.0 * corr_upto
        C_map = C_now.reshape(H, W)

        plot_path = os.path.join(plots_dir, f"cmap_upto_{k}.png")
        if not os.path.exists(plot_path):
            save_cmap_plot(
                img, C_map, beta, kappa, k,
                plot_path,
                title_prefix=f"{prune_mode}-win{window}",
                active_mask=active_mask.reshape(H, W),
                frozen_sz=frozen_sz.reshape(H, W),
            )
            print(f"[upto k={k}] saved plot -> {plot_path}")
        else:
            print(f"[upto k={k}] plot exists; skipping")

        # pruning
        if (k >= k_freeze_start) and (current_n > 0) and ((window - 1) > 0):
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
            nf = int(freeze.sum())
        else:
            freeze = np.zeros(n0, dtype=bool)
            nf = 0

        print(f"[prune] freeze candidates = {nf}")

        if nf > 0:
            idxs = np.where(freeze)[0]
            frozen_sz[idxs] = np.where(C_now[idxs] >= 0.0, 0.5, -0.5)
            active_mask[idxs] = False
            edges, active_sites = remap_edges(edges0, active_mask, n0)
            current_n = len(active_sites)
            print(f"[prune] new active spins = {current_n}")

        # update history
        if len(C_hist) > 0:
            C_hist = [C_now] + C_hist[:-1]

        # state checkpoint
        ckpt_out = os.path.join(ckpt_dir, f"state_k_{k}.npz")
        edges_ij, edges_Jz = pack_edges(edges)
        atomic_savez(
            ckpt_out,
            active_mask=active_mask.astype(np.uint8),
            frozen_sz=frozen_sz.astype(np.float64),
            edges_ij=edges_ij,
            edges_Jz=edges_Jz,
            current_n=np.array(current_n, dtype=np.int64),
            Z_cum=np.array(Z_cum, dtype=np.float64),
            Num_cum=Num_cum.astype(np.float64),
            C_hist=np.stack(C_hist, axis=0).astype(np.float64) if len(C_hist) > 0 else np.zeros((0, n0), dtype=np.float64),
            window=np.array(window, dtype=np.int64),
        )
        print(f"[checkpoint] saved -> {ckpt_out}")

        if current_n == 0:
            print("All spins frozen. Continuing sectors without further pruning.")

    # final map (after finishing all k up to k_top)
    corr_upto = Num_cum / Z_cum
    C_final = 2.0 * corr_upto
    C_final_map = C_final.reshape(H, W)

    final_path = os.path.join(out_dir, "final_correlation_map.png")
    save_cmap_plot(
        img, C_final_map, beta, kappa, k="final",
        outpath=final_path,
        title_prefix=f"FINAL {prune_mode}-win{window}",
        active_mask=active_mask.reshape(H, W),
        frozen_sz=frozen_sz.reshape(H, W),
    )
    print(f"[final] saved -> {final_path}")

    # final classification map from final correlations
    TAU_CLASS = 0.0   # <- increase (e.g. 0.1, 0.2) to create an "unknown" band
    cls_final = classify_from_C(C_final_map, tau_class=TAU_CLASS,
                                pos_value=255, neg_value=0, unk_value=127)

    cls_path = os.path.join(out_dir, "final_classification_map.png")
    save_classification_plot(
        img, cls_final,
        outpath=cls_path,
        title=f"FINAL classification (tau_class={TAU_CLASS})",
        active_mask=active_mask
    )
    print(f"[final] classification saved -> {cls_path}")

    print("\nDone. Output in:", out_dir)


# -------------------------
# Example (12x12 MNIST)
# -------------------------

# if __name__ == "__main__":

#     img = load_mnist_12x12(index=0)  # change index for different digit

#     H, W = img.shape

#     beta = 6.0
#     kappa = 3.0

#     # Keep k small or this explodes.
#     k_top = 4

#     WINDOW = 3
#     PRUNE_MODE = "stability_only"
#     CAP_FRAC = 1.0
#     EPS = 0.005

#     out_dir = (
#         f"out_pruned_"
#         f"{H}x{W}_"
#         f"MNIST_"
#         f"{PRUNE_MODE}_"
#         f"win{WINDOW}_"
#         f"eps{EPS}"
#     )

#     run_resumable(
#         img,
#         beta=beta,
#         kappa=kappa,
#         k_top=k_top,
#         label_site=0,
#         n_samples=32,   # reduce for speed
#         seed=0,
#         k_freeze_start=3,
#         eps=EPS,
#         tau=0.15,
#         tau_B=0.05,
#         min_agree=3,
#         prune_mode=PRUNE_MODE,
#         cap_frac=CAP_FRAC,
#         window=WINDOW,
#         out_dir=out_dir
#     )

if __name__ == "__main__":
    H = 5
    W = 5

    img = make_plus_image(H=H, W=W, white=255.0, black=0.0, thickness=1)

    beta = 10.0
    kappa = 3.0
    k_top = 9

    WINDOW = 3
    PRUNE_MODE = "stability_only"
    CAP_FRAC = 1.0

    # 🔹 Automatic output directory
    out_dir = (
        f"out_pruned_"
        f"{H}x{W}_"
        f"{PRUNE_MODE}_"
        f"win{WINDOW}_"
        f"eps{0.001}"
    )

    run_resumable(
        img,
        beta=beta,
        kappa=kappa,
        k_top=k_top,
        label_site=0,
        n_samples=64,
        seed=0,
        k_freeze_start=3,
        eps=0.005,
        tau=0.20,
        tau_B=0.06,
        min_agree=3,
        prune_mode=PRUNE_MODE,
        cap_frac=CAP_FRAC,
        window=WINDOW,
        out_dir=out_dir
    )