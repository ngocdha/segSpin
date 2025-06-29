import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
import time

start_time = time.time()

# === 1. Input Image ===
n = 12
img = np.concatenate([255 * np.ones(4), np.ones(4), 255 * np.ones(4)])

# === 2. Spin Operators and Identity ===
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, 1j], [-1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

# === 3. Pixel Similarity → Coupling Strengths J(i) ===
J = np.zeros(n)
for i in range(n-1):
    diff = abs(img[i] - img[i+1])
    J[i] = 2 - (4/256) * diff

# === 4. Construct Ising-like Hamiltonian (dense version) ===
H = np.zeros((2**n, 2**n), dtype=complex)

# (a) Nearest-neighbor interactions
for i in range(n-1):
    opX = 1
    opY = 1
    opZ = 1
    for j in range(n):
        if j == i or j == i+1:
            X, Y, Z = sx, sy, sz
        else:
            X, Y, Z = I, I, I
        opX = np.kron(opX, X)
        opY = np.kron(opY, Y)
        opZ = np.kron(opZ, Z)
    H -= (opY + opZ) + J[i] * opX

# (b) Label constraint: only one label at site 2 (+1)
label_strength = 1
label_index = 1  # 1-based index becomes 0-based

op = 1
for j in range(n):
    A = sx if j == label_index else I
    op = np.kron(op, A)
H -= label_strength * (+1) * op

# === 5. Solve for Ground State ===
eigenvalues, eigenvectors = eig(H)
idx = np.argmin(eigenvalues.real)
ground_state = eigenvectors[:, idx]

# === 6. Measure ⟨σx⟩ for Each Pixel ===
expect_vals = np.zeros(n)
for i in range(n):
    op = 1
    for j in range(n):
        A = sx if j == i else I
        op = np.kron(op, A)
    expect_vals[i] = np.real(np.conj(ground_state).T @ (op @ ground_state))

# === 7. Assign Segments ===
segments = expect_vals >= 0

# === 9. Plotting ===
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
