import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

# -------------------------------
# Load all results into a DataFrame
# -------------------------------
files = sorted(glob("results/energies_task*.npz"))
rows = []

for f in files:
    d = np.load(f)
    rows.append({
        "task_id":     int(d["task_id"]),
        "digit":       int(d["digit"]),
        "mnist_idx":   int(d["mnist_idx"]),
        "is_black":    int(d["is_black"]),   # 1 if black image
        "E_min":       float(d["E_min"]),
        "E_max":       float(d["E_max"]),
    })

df = pd.DataFrame(rows)

print(df.head())
print("\nBlack image row:")
print(df[df["is_black"] == 1])

# Identify black image values
E_min_black = df[df["is_black"] == 1]["E_min"].iloc[0]
E_max_black = df[df["is_black"] == 1]["E_max"].iloc[0]


# -------------------------------
# Plot histogram of E_min
# -------------------------------
plt.figure(figsize=(8,5))
plt.hist(df["E_min"], bins=30, alpha=0.7, color="steelblue", edgecolor="black")
plt.axvline(E_min_black, color="red", linestyle="--", linewidth=2,
            label=f"Black image E_min = {E_min_black:.4f}")

plt.title("Histogram of Ground-State Energy E_min (H)")
plt.xlabel("E_min")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("hist_E_min.png", dpi=150)
plt.close()


# -------------------------------
# Plot histogram of E_max
# -------------------------------
plt.figure(figsize=(8,5))
plt.hist(df["E_max"], bins=30, alpha=0.7, color="darkgreen", edgecolor="black")
plt.axvline(E_max_black, color="red", linestyle="--", linewidth=2,
            label=f"Black image E_max = {E_max_black:.4f}")

plt.title("Histogram of Highest Energy E_max (H)")
plt.xlabel("E_max")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("hist_E_max.png", dpi=150)
plt.close()

print("Saved hist_E_min.png and hist_E_max.png")
