#!/usr/bin/env python3
"""
scaling_analysis_from_table.py

Analyze scaling of the spectral width (gap = E_max - E_min)
using the pre-made CSV table: eigenvalue_summary_all_sizes.csv

This table must have at least the columns:
    nr, digit, is_black, E_min, E_max
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# 1. Load Table
# ============================

CSV_FILE = "eigenvalue_summary.csv"
df = pd.read_csv(CSV_FILE)

required_cols = ["nr", "digit", "is_black", "E_min", "E_max"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' missing from CSV.")

print("Loaded table with", len(df), "rows.")
print("Unique sizes:", sorted(df["nr"].unique()))

# Compute gap
df["gap"] = df["E_max"] - df["E_min"]

# Save updated table (optional)
df.to_csv("eigenvalue_summary_with_gap.csv", index=False)
print("Saved updated table: eigenvalue_summary_with_gap.csv")


# ============================
# 2. Scaling of mean gap vs nr (MNIST only)
# ============================

df_mnist = df[df["is_black"] == 0].copy()   # ignore black image for “generic MNIST”
group = df_mnist.groupby("nr")["gap"]

mean_gap = group.mean()
std_gap  = group.std()
nr_vals  = np.array(sorted(mean_gap.index), dtype=float)
gap_vals = mean_gap.values

print("\n=== Mean gap vs size (MNIST only) ===")
for n, g, s in zip(nr_vals, gap_vals, std_gap.values):
    print(f"nr={int(n)}: mean gap={g:.6f} ± {s:.6f}")

# Power-law fit on log–log scale
log_n = np.log(nr_vals)
log_gap = np.log(gap_vals)

alpha, log_A = np.polyfit(log_n, log_gap, 1)
A = np.exp(log_A)

print("\n=== Power-law fit (MNIST) ===")
print("Fit: gap(n) ≈ A n^alpha")
print(f"A ≈ {A:.6e}")
print(f"alpha ≈ {alpha:.6f}")


# ============================
# 3. Per-digit scaling
# ============================

print("\n=== Per-digit scaling exponents ===")
digit_results = []
for digit in sorted(df_mnist["digit"].unique()):
    df_d = df_mnist[df_mnist["digit"] == digit]
    group_d = df_d.groupby("nr")["gap"]

    if group_d.ngroups < 3:
        continue  # need at least 3 data points

    n_d = np.array(sorted(group_d.mean().index), dtype=float)
    gap_d = group_d.mean().values

    log_n_d = np.log(n_d)
    log_gap_d = np.log(gap_d)

    alpha_d, log_A_d = np.polyfit(log_n_d, log_gap_d, 1)
    A_d = np.exp(log_A_d)

    digit_results.append((digit, A_d, alpha_d))
    print(f"Digit {digit}: A ≈ {A_d:.6e}, alpha ≈ {alpha_d:.6f}")


# ============================
# 4. Black-image scaling
# ============================

df_black = df[df["is_black"] == 1].sort_values("nr")

print("\n=== Black image gaps ===")
for _, r in df_black.iterrows():
    print(f"nr={int(r['nr'])}: gap={r['gap']:.6f}, "
          f"E_min={r['E_min']:.6f}, E_max={r['E_max']:.6f}")


# ============================
# 5. Plot: log–log scaling curve
# ============================

plt.figure(figsize=(7,5))

# MNIST mean gap (points)
plt.loglog(nr_vals, gap_vals, "o", label="MNIST mean gap")

# Fit line
n_fit = np.linspace(nr_vals.min(), nr_vals.max(), 200)
gap_fit = A * n_fit**alpha
plt.loglog(n_fit, gap_fit, "-", label=f"Fit: n^{alpha:.2f}")

# Black image (squares)
if not df_black.empty:
    plt.loglog(df_black["nr"], df_black["gap"], "s", label="Black image")

plt.xlabel("Image size n")
plt.ylabel("Gap = E_max - E_min")
plt.title("Spectral width scaling (log–log)")
plt.legend()
plt.tight_layout()
plt.savefig("gap_scaling_loglog.png", dpi=150)
plt.close()
print("Saved: gap_scaling_loglog.png")

# ============================
# 6. Linear-scale diagnostic plot
# ============================

plt.figure(figsize=(7,5))
plt.errorbar(nr_vals, gap_vals, yerr=std_gap.values,
             fmt="o-", capsize=3, label="MNIST mean")
plt.plot(df_black["nr"], df_black["gap"], "s-", label="Black image")
plt.xlabel("Image size n")
plt.ylabel("Gap = E_max - E_min")
plt.title("Spectral width vs image size (linear)")
plt.legend()
plt.tight_layout()
plt.savefig("gap_scaling_linear.png", dpi=150)
plt.close()
print("Saved: gap_scaling_linear.png")

print("\nDone.")
