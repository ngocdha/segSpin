#!/usr/bin/env python3
"""
mnist_dmrg_tenpy.py

Ground-state Sz_l Sz_i correlator for an image-based XXZ model using TeNPy.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

import tenpy
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

from torchvision import datasets, transforms
from skimage.transform import resize


# Utilities:

def image_to_edges2d(img, kappa=2.0):
    """
    Given a 2D grayscale image (numpy array) in [0, 255],
    return a list of (i, j, Jz_ij) couplings, using the same formula
    you had in Julia:

        Jz_ij = 2.0 - (kappa / sigma) * |I_i - I_j|

    where sigma is the global std of intensities.
    """
    nrows, ncols = img.shape
    grid = img.astype(float)
    sigma = grid.std()
    if sigma <= 0:
        sigma = 1.0

    edges = []
    for r in range(nrows):
        for c in range(ncols):
            i = r * ncols + c  # 0-based index
            # right neighbor
            if c + 1 < ncols:
                diff = abs(grid[r, c] - grid[r, c + 1])
                Jz = 2.0 - (kappa / sigma) * diff
                edges.append((i, i + 1, Jz))
            # down neighbor
            if r + 1 < nrows:
                diff = abs(grid[r, c] - grid[r + 1, c])
                j = (r + 1) * ncols + c
                Jz = 2.0 - (kappa / sigma) * diff
                edges.append((i, j, Jz))
    return edges, nrows, ncols


# Custom TeNPy model: image-based XXZ

class ImageXXZModel(CouplingMPOModel):
    """
    H = sum_edges [ -Sx_i Sx_j - Sy_i Sx_j - Jz_ij Sz_i Sz_j ]
    implemented via S+, S-:
        Sx Sx + Sy Sy = 0.5 (Sp Sm + Sm Sp)
    """

    def init_sites(self, model_params):
        conserve = model_params.get("conserve", None)
        site = SpinHalfSite(conserve=conserve)
        return site  # IMPORTANT: return the Site, not (site,)

    def init_terms(self, model_params):
        edges = model_params["edges"]
        Jxy = model_params.get("Jxy", -1.0)

        for (i, j, Jz) in edges:
            if j < i:
                i, j = j, i
            # -SxSx - SySy = Jxy * 0.5 (SpSm + SmSp) with Jxy=-1
            self.add_coupling_term(0.5 * Jxy, i, j, "Sp", "Sm")
            self.add_coupling_term(0.5 * Jxy, i, j, "Sm", "Sp")
            # -Jz SzSz
            self.add_coupling_term(-Jz, i, j, "Sz", "Sz")



# Main: DMRG ground state correlator

def main():
    if len(sys.argv) < 3:
        print("Usage: python mnist_dmrg_tenpy.py <nr> <kappa>")
        sys.exit(1)

    nr = int(sys.argv[1])
    kappa = float(sys.argv[2])
    print(f"Running TeNPy DMRG for nr={nr}, kappa={kappa}")

    tenpy.tools.misc.setup_logging(to_stdout="INFO")

    # ---- Load a single MNIST digit (e.g. digit 8) ----
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root="./data", train=True,
                                 download=True, transform=transform)

    target_digit = 8
    idx = next(i for i, (_, lab) in enumerate(mnist_train) if lab == target_digit)
    img_tensor, label = mnist_train[idx]        # shape (1, 28, 28)
    img28 = img_tensor.squeeze(0).numpy()       # (28, 28), in [0, 1]
    img28 = img28 * 255.0                       # match Julia scaling

    # ---- Resize to (nr, nr) ----
    if nr != 28:
        img_small = resize(img28, (nr, nr), order=1,
                           mode="reflect", anti_aliasing=True)
        # bring back to roughly [0, 255]
        img_small *= 255.0 / img_small.max()
    else:
        img_small = img28.copy()

    # ---- Build edges / couplings ----
    edges, nrows, ncols = image_to_edges2d(img_small, kappa=kappa)
    L = nrows * ncols

    print(f"Image size: {nrows} x {ncols}, L = {L}, #edges = {len(edges)}")

    # ---- Define TeNPy model parameters ----
    model_params = dict(
        L=L,
        lattice="Chain",      # 1D chain
        bc_MPS="finite",
        bc_x="open",
        conserve=None,
        edges=edges,
        Jxy=-1.0,             # gives -SxSx - SySy
    )

    model = ImageXXZModel(model_params)

    # ---- Initial MPS: product state |↑↑…↑> ----
    product_state = ["up"] * model.lat.N_sites  # length L
    psi = MPS.from_product_state(
        model.lat.mps_sites(),  # list of Site objects
        product_state,          # ["up", "up", ..., "up"]
        bc=model.lat.bc_MPS
    )

    # ---- DMRG ground state ----
    dmrg_params = {
        "mixer": None,
        "trunc_params": {
            "chi_max": 100,
            "svd_min": 1e-10,
        },
        "max_E_err": 1e-8,
        "min_sweeps": 2,    # optional
        "max_sweeps": 10,   # do up to 10 sweeps
        # "N_sweeps_check": 1,  # optional: check convergence every sweep
    }


    eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E0, psi = eng.run()
    print(f"Ground-state energy E0 = {E0}")

    # ---- Ground-state ⟨Sz_l Sz_i⟩ correlator ----
    # choose center pixel as label site
    r0 = nrows // 2
    c0 = ncols // 2
    ell = r0 * ncols + c0   # 0-based index

    # correlation_function('Sz', 'Sz') returns a matrix C[i,j] = <Sz_i Sz_j>
    C = psi.correlation_function("Sz", "Sz")
    corr_vec = np.real(C[ell, :])   # row corresponding to site ell
    corr_map = corr_vec.reshape(nrows, ncols)

    # ---- Plot: input vs correlator ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(np.flipud(img_small), cmap="gray")
    axes[0].set_title(f"MNIST digit {label} ({nr}x{nr})")
    axes[0].axis("off")

    axes[1].imshow(np.flipud(corr_map), cmap="viridis")
    axes[1].set_title(r"Ground-state $\langle S^z_\ell S^z_i \rangle$")
    axes[1].axis("off")

    plt.tight_layout()
    outfile = f"mnist_{label}_{nr}x{nr}_kappa{kappa}.png"
    plt.savefig(f"plots/{outfile}", dpi=150)
    print(f"Saved figure to plots/{outfile}")


if __name__ == "__main__":
    main()
