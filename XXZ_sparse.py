import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import eigsh
import time

start_time = time.time()

# === 1. Input Image (generalized) ===
n = 25  # total number of pixels

bright = 255
dark = 1

bright_len1 = n // 3
dark_len = n // 3
bright_len2 = n - bright_len1 - dark_len

img = np.concatenate([
    bright * np.ones(bright_len1),
    dark * np.ones(dark_len),
    bright * np.ones(bright_len2)
])

# Pauli matrices (sparse)
sx = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
sy = csr_matrix(np.array([[0, 1j], [-1j, 0]], dtype=complex))
sz = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
I2 = identity(2, format='csr', dtype=complex)

# === 2. Compute similarity-based couplings J ===
J = np.zeros(n)
for i in range(n-1):
    diff = abs(img[i] - img[i+1])
    J[i] = 2 - (4/256) * diff

# === 3. Precompute local operators ===
def build_local(op, site):
    ops = [I2] * n
    ops[site] = op
    result = ops[0]
    for k in range(1, n):
        result = kron(result, ops[k], format='csr')
    return result

Sx = [build_local(sx, i) for i in range(n)]
Sy = [build_local(sy, i) for i in range(n)]
Sz = [build_local(sz, i) for i in range(n)]

# === 4. Build Hamiltonian ===
H = csr_matrix((2**n, 2**n), dtype=complex)

# (a) Nearest-neighbor interactions
for i in range(n-1):
    H -= (Sy[i] @ Sy[i+1] + Sz[i] @ Sz[i+1] + J[i] * Sx[i] @ Sx[i+1])

# (b) Label constraints (general structure for future multiple seeds)
label_strength = 1
label_sites = [2] 
label_targets = [+1]

for site, target in zip(label_sites, label_targets):
    H -= label_strength * target * Sx[site]

# === 5. Solve ground state ===
eigval, eigvec = eigsh(H, k=1, which='SA')
ground_state = eigvec[:, 0]

# === 6. Measure ⟨σx⟩ for each site ===
expect_vals = np.zeros(n)
for i in range(n):
    expect_vals[i] = np.real(np.vdot(ground_state, Sx[i] @ ground_state))

segments = expect_vals >= 0

# === 7. Plotting ===
plt.figure(figsize=(8, 10))

plt.subplot(3, 1, 1)
plt.bar(range(1, n + 1), img)
plt.title("Input 1D Image (Pixel Intensities)")
plt.ylabel("Intensity")
plt.grid()

plt.subplot(3, 1, 2)
plt.bar(range(1, n + 1), expect_vals, color=[0.2, 0.6, 0.8])
plt.title("⟨σ_x⟩ Values (Spin State)")
plt.ylabel("⟨σ_x⟩")
plt.axhline(0, linestyle='--', color='k')
plt.grid()

plt.subplot(3, 1, 3)
plt.bar(range(1, n + 1), segments.astype(int), color=[0.9, 0.5, 0.2])
plt.title("Segment Assignment (1 = Foreground, 0 = Background)")
plt.xlabel("Pixel Index")
plt.ylabel("Segment")
plt.ylim(-0.5, 1.5)
plt.grid()

plt.tight_layout()
plt.show()

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: %.2f seconds" % elapsed_time)


# make sure Hamiltonian is correct
# Tune parameters to get 2 segments