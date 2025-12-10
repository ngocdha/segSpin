#!/usr/bin/env python3
"""
plot_segmentation_maps.py

Plot "segmentation maps" derived from Z-spin correlations for
ground state and 10th excited state for a given (nr, img_id).

Uses files produced by MNIST_dmrg_corr_excited.py:
    results_corr/corr_nr{nr}_img{img_id}.npz

Segmentation logic:
- choose center site ℓ (middle pixel)
- take row C[ℓ, :] from C_gs and C_exc10
- reshape to (nr, nr)
- form binary masks (corr > 0) as segmentation maps
- plot original image + both segmentations (+ correlation heatmaps)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from skimage.transform import resize


def load_corr_file(nr, img_id, base_dir="results_corr"):
    fname = os.path.join(base_dir, f"corr_nr{nr}_img{img_id}.npz")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Could not find {fname}")
    data = np.load(fname, allow_pickle=True)
    return data


def reconstruct_image(nr, digit, mnist_idx):
    """
    Rebuild the nr×nr image from MNIST index and digit label.
    (nr is just for resizing; digit is for sanity checks only.)
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"]
    y = mnist["target"].astype(int)

    # sanity check: the label for mnist_idx should match `digit`
    if int(y[mnist_idx]) != int(digit):
        print(f"Warning: digit label mismatch: stored digit={digit}, "
              f"MNIST[{mnist_idx}].y={y[mnist_idx]}")

    img_flat = X[mnist_idx].astype(np.float32)
    img28 = img_flat.reshape(28, 28)

    if nr != 28:
        img_small = resize(
            img28, (nr, nr),
            order=1,
            mode="reflect",
            anti_aliasing=True,
        )
        m = img_small.max()
        if m > 0:
            img_small *= 255.0 / m
    else:
        img_small = img28.copy()

    return img_small.astype(np.float32)


def make_segmentation_maps(C_gs, C_exc10, nrows, ncols):
    """
    Given two full correlation matrices (L x L),
    extract the center-site row and reshape to (nrows, ncols).
    Then make binary segmentation masks based on > 0 correlation.
    """
    L = nrows * ncols
    assert C_gs.shape == (L, L)
    assert C_exc10.shape == (L, L)

    # center pixel (row-major flattening)
    r0 = nrows // 2
    c0 = ncols // 2
    ell = r0 * ncols + c0

    corr_gs = np.real(C_gs[ell, :]).reshape(nrows, ncols)
    corr_exc = np.real(C_exc10[ell, :]).reshape(nrows, ncols)

    # very simple segmentation: positive vs non-positive correlation
    seg_gs = (corr_gs > 0).astype(float)
    seg_exc = (corr_exc > 0).astype(float)

    return corr_gs, corr_exc, seg_gs, seg_exc


def rot180(arr):
    """Rotate array by 180 degrees (flip both axes)."""
    return np.flipud(np.fliplr(arr))


def plot_segmentation(img, corr_gs, corr_exc, seg_gs, seg_exc,
                      nr, img_id, digit, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    # Apply 180° rotation so orientation matches your reference
    img_plot      = rot180(img)
    corr_gs_plot  = rot180(corr_gs)
    corr_exc_plot = rot180(corr_exc)
    seg_gs_plot   = rot180(seg_gs)
    seg_exc_plot  = rot180(seg_exc)

    # 2x3 layout:
    # [0,0] original image
    # [0,1] GS correlation
    # [0,2] 10th-exc correlation
    # [1,0] (unused, hidden)
    # [1,1] GS segmentation
    # [1,2] 10th-exc segmentation
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))

    # original image
    ax = axes[0, 0]
    ax.imshow(img_plot, cmap="gray")
    ax.set_title(f"Original digit {digit}, nr={nr}, img_id={img_id}")
    ax.axis("off")

    # GS correlation heatmap
    ax = axes[0, 1]
    im1 = ax.imshow(corr_gs_plot, cmap="viridis")
    ax.set_title(r"GS $C_{\ell i} = \langle S^z_\ell S^z_i \rangle$")
    ax.axis("off")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    # 10th-exc correlation heatmap
    ax = axes[0, 2]
    im2 = ax.imshow(corr_exc_plot, cmap="viridis")
    ax.set_title(r"10th exc. $C_{\ell i}$")
    ax.axis("off")
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # hide bottom-left panel (no segmentation difference)
    axes[1, 0].axis("off")
    axes[1, 0].set_visible(False)

    # GS segmentation map
    ax = axes[1, 1]
    ax.imshow(seg_gs_plot, cmap="gray", vmin=0, vmax=1)
    ax.set_title("GS segmentation (corr > 0)")
    ax.axis("off")

    # 10th-exc segmentation map
    ax = axes[1, 2]
    ax.imshow(seg_exc_plot, cmap="gray", vmin=0, vmax=1)
    ax.set_title("10th-exc segmentation (corr > 0)")
    ax.axis("off")

    fig.suptitle("Center-spin Z-correlation segmentation maps", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    outfile = os.path.join(out_dir, f"seg_nr{nr}_img{img_id}.png")
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved segmentation figure: {outfile}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_segmentation_maps.py <nr> <img_id>")
        print("  nr in {4,5,6,7,8,9}")
        print("  img_id in {0..49}")
        sys.exit(1)

    nr = int(sys.argv[1])
    img_id = int(sys.argv[2])

    data = load_corr_file(nr, img_id)
    digit = int(data["digit"])
    mnist_idx = int(data["mnist_idx"])
    nrows = int(data["nrows"])
    ncols = int(data["ncols"])

    C_gs = np.array(data["C_gs"])
    C_exc10 = np.array(data["C_exc10"])

    img = reconstruct_image(nr, digit, mnist_idx)
    corr_gs, corr_exc, seg_gs, seg_exc = make_segmentation_maps(
        C_gs, C_exc10, nrows, ncols
    )
    plot_segmentation(img, corr_gs, corr_exc, seg_gs, seg_exc,
                      nr, img_id, digit)


if __name__ == "__main__":
    main()
