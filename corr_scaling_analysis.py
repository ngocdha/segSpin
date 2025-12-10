#!/usr/bin/env python3
"""
corr_scaling_analysis.py

Analyze how the differences between Z-spin correlations
of the ground state and the 10th excited state scale
with image size (nr).

Assumes files:
    results_corr/corr_nr{nr}_img{img_id}.npz

produced by MNIST_dmrg_corr_excited.py, each containing:
    nr, img_id, digit, mnist_idx, kappa, sigma, nrows, ncols, L,
    E0, E10, C_gs, C_exc10
"""

import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import os

# -------------------------------
# 1. Load all correlation files
# -------------------------------

files = sorted(glob("results_corr/corr_nr*_img*.npz"))
if not files:
    raise RuntimeError("No files found matching 'results_corr/corr_nr*_img*.npz'")

rows = []
for fname in files:
    data = np.load(fname, allow_pickle=True)

    nr       = int(data["nr"])
    img_id   = int(data["img_id"])
    digit    = int(data["digit"])
    mnist_idx = int(data["mnist_idx"])
    kappa    = float(data["kappa"])
    sigma    = float(data["sigma"])
    nrows    = int(data["nrows"])
    ncols    = int(data["ncols"])
    L        = int(data["L"])
    E0       = float(data["E0"])
    E10      = float(data["E10"])

    C_gs    = np.array(data["C_gs"])
    C_exc10 = np.array(data["C_exc10"])
    assert C_gs.shape == C_exc10.shape, "C_gs and C_exc10 shape mismatch"
    assert C_gs.shape == (L, L), f"Unexpected correlation shape {C_gs.shape} for L={L}"

    # -------------------------------
    # 2. Compute difference metrics
    #    ΔC = C_exc10 - C_gs
    # -------------------------------
    dC = C_exc10 - C_gs

    # Global metrics
    fro_norm   = np.linalg.norm(dC)               # Frobenius norm
    mad        = np.mean(np.abs(dC))              # mean abs diff
    max_abs    = np.max(np.abs(dC))               # max abs diff

    # Central site metrics (correlations from "label" spin)
    # Choose center pixel: (r0, c0)
    r0 = nrows // 2
    c0 = ncols // 2
    ell = r0 * ncols + c0   # 0-based index in flattened chain

    dC_row = dC[ell, :]
    row_fro_norm = np.linalg.norm(dC_row)
    row_mad      = np.mean(np.abs(dC_row))
    row_max_abs  = np.max(np.abs(dC_row))

    rows.append(
        dict(
            filename=os.path.basename(fname),
            nr=nr,
            img_id=img_id,
            digit=digit,
            mnist_idx=mnist_idx,
            kappa=kappa,
            sigma=sigma,
            nrows=nrows,
            ncols=ncols,
            L=L,
            E0=E0,
            E10=E10,
            gap_E=E10 - E0,
            fro_norm=fro_norm,
            mad=mad,
            max_abs=max_abs,
            row_fro_norm=row_fro_norm,
            row_mad=row_mad,
            row_max_abs=row_max_abs,
        )
    )

df = pd.DataFrame(rows)
df.to_csv("corr_diff_summary.csv", index=False)
print("Saved table: corr_diff_summary.csv")
print("First few rows:")
print(df.head())


# -------------------------------
# 3. Scaling analysis helpers
# -------------------------------

def fit_power_law(n_vals, y_vals, label):
    """
    Fit y ~ A * n^alpha using log-log regression.
    Returns (A, alpha).
    """
    n_vals = np.asarray(n_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)

    # filter out non-positive y (log undefined)
    mask = y_vals > 0
    n_vals = n_vals[mask]
    y_vals = y_vals[mask]
    if len(n_vals) < 2:
        raise RuntimeError(f"Not enough positive data points to fit power law for {label}")

    log_n = np.log(n_vals)
    log_y = np.log(y_vals)

    alpha, log_A = np.polyfit(log_n, log_y, 1)
    A = np.exp(log_A)
    print(f"\n=== Power-law fit for {label} ===")
    print(f"Fit: {label}(n) ≈ A * n^alpha")
    print(f"A ≈ {A:.6e}, alpha ≈ {alpha:.6f}")
    return A, alpha


# -------------------------------
# 4. Mean metrics vs nr
# -------------------------------

metrics = ["fro_norm", "mad", "max_abs", "row_fro_norm", "row_mad", "row_max_abs"]
means = df.groupby("nr")[metrics].mean()
stds  = df.groupby("nr")[metrics].std()
n_values = np.array(sorted(df["nr"].unique()), dtype=float)

print("\n=== Mean correlation-difference metrics vs size ===")
print(means)

means.to_csv("corr_diff_means_by_nr.csv")
stds.to_csv("corr_diff_stds_by_nr.csv")
print("\nSaved: corr_diff_means_by_nr.csv, corr_diff_stds_by_nr.csv")

# -------------------------------
# 5. Fit power laws for key metrics
# -------------------------------

# Use global Frobenius norm and central-row Frobenius norm as primary
fro_means      = means["fro_norm"].loc[n_values]
row_fro_means  = means["row_fro_norm"].loc[n_values]

A_fro, alpha_fro = fit_power_law(n_values, fro_means, "||ΔC||_F (global)")
A_row, alpha_row = fit_power_law(n_values, row_fro_means, "||ΔC_row||_F (center site)")


# -------------------------------
# 6. Plots: log-log scaling curves
# -------------------------------

plt.figure(figsize=(7,5))
plt.loglog(n_values, fro_means, "o-", label="mean ||ΔC||_F (global)")
plt.loglog(n_values, row_fro_means, "s-", label="mean ||ΔC_row||_F (center)")
# fit lines
n_fit = np.linspace(n_values.min(), n_values.max(), 200)
plt.loglog(n_fit, A_fro * n_fit**alpha_fro, "--", label=f"fit global, α={alpha_fro:.2f}")
plt.loglog(n_fit, A_row * n_fit**alpha_row, "--", label=f"fit center, α={alpha_row:.2f}")

plt.xlabel("Image size n (nr)")
plt.ylabel("Correlation difference metric")
plt.title("Scaling of ground vs 10th-excited Z-spin correlations (log-log)")
plt.legend()
plt.tight_layout()
plt.savefig("corr_diff_scaling_loglog.png", dpi=150)
plt.close()
print("Saved: corr_diff_scaling_loglog.png")

# -------------------------------
# 7. Optional: linear-scale diagnostic
# -------------------------------

plt.figure(figsize=(7,5))
plt.errorbar(n_values, fro_means, yerr=stds["fro_norm"].loc[n_values],
             fmt="o-", capsize=3, label="mean ||ΔC||_F (global)")
plt.errorbar(n_values, row_fro_means, yerr=stds["row_fro_norm"].loc[n_values],
             fmt="s-", capsize=3, label="mean ||ΔC_row||_F (center)")

plt.xlabel("Image size n (nr)")
plt.ylabel("Correlation difference metric")
plt.title("Correlation differences vs image size (linear axes)")
plt.legend()
plt.tight_layout()
plt.savefig("corr_diff_scaling_linear.png", dpi=150)
plt.close()
print("Saved: corr_diff_scaling_linear.png")

print("\nDone.")
