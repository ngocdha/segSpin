import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import eigsh
import time
from PIL import Image

start_time = time.time()

# === 1. Input Image (generalized) ===
n = 4  # number of rows
m = 4  # number of columns
nt = n*m

img = Image.open("mario.jpg")
img_gray = img.convert("L")
img_array = np.array(img_gray)

# Break into 4x4 blocks
blocks = img_array.reshape(4, 4, 4, 4).swapaxes(1, 2)
block_means = np.array([[np.mean(block) for block in row] for row in blocks])

img = block_means.reshape(nt,1)
img2 = [el[0] for el in img]
img = img2

# Pauli matrices (sparse)
sx = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
sy = csr_matrix(np.array([[0, 1j], [-1j, 0]], dtype=complex))
sz = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
I2 = identity(2, format='csr', dtype=complex)

# === 2. Compute similarity-based couplings J ===
Jh = np.zeros(nt)
for i in range(nt-1):
    print(i)
    diff = abs(img[i] - img[i+1])
    Jh[i] = (2 - (8/256) * diff)
    
Jv = np.zeros(nt)
for i in range(nt-m):
    diff = abs(img[i] - img[i+m])
    Jv[i] = (2 - (8/256) * diff)
    
# === 3. Precompute local operators ===
def build_local(op, site):
    ops = [I2] * nt
    ops[site] = op
    result = ops[0]
    for k in range(1, nt):
        result = kron(result, ops[k], format='csr')
    return result

Sx = [build_local(sx, i) for i in range(nt)]
Sy = [build_local(sy, i) for i in range(nt)]
Sz = [build_local(sz, i) for i in range(nt)]

# === 4. Build Hamiltonian ===
H = csr_matrix((2**nt, 2**nt), dtype=complex)

# (a) Nearest-neighbor interactions
for i in range(nt-1):
    if i%m != m-1:
        H -= (Sy[i] @ Sy[i+1] + Sz[i] @ Sz[i+1] + Jh[i] * Sx[i] @ Sx[i+1])
        
for i in range(nt-m):
    H -= (Sy[i] @ Sy[i+m] + Sz[i] @ Sz[i+m] + Jv[i] * Sx[i] @ Sx[i+m])
    
# (b) Label constraints
label_strength = 1
label_sites = [11] 
label_targets = [+1]

for site, target in zip(label_sites, label_targets):
    H -= label_strength * target * Sx[site]
    
# === 5. Solve ground state ===
eigval, eigvec = eigsh(H, k=1, which='SA')
ground_state = eigvec[:, 0]

# === 6. Measure ⟨σx⟩ for each site ===
expect_vals = np.zeros(nt)
for i in range(nt):
    expect_vals[i] = np.real(np.vdot(ground_state, Sx[i] @ ground_state))

segments = expect_vals >= 0

# === 7. Plotting ===
# Reshape for 2D visualization
expect_vals_2d = expect_vals.reshape(n, m)
segments_2d = segments.reshape(n, m)
img_2d = np.array(img).reshape(n, m)

# Compute (row, col) coordinates of label_sites
label_coords = []
for site in label_sites:
    row = site // m
    col = site % m
    label_coords.append( (row, col) )

# Plot
plt.figure(figsize=(12, 4))

# Plot original block-averaged image
plt.subplot(1, 3, 1)
plt.imshow(img_2d, cmap='gray')
plt.title("Input Block-Averaged Image")
plt.colorbar()
# Draw seed marker
for (row, col) in label_coords:
    plt.scatter(col, row, s=200, facecolors='none', edgecolors='red', linewidths=2)

# Plot ⟨σ_x⟩ values as heatmap
plt.subplot(1, 3, 2)
plt.imshow(expect_vals_2d, cmap='coolwarm')
plt.title("⟨σ_x⟩ Values Heatmap")
plt.colorbar()

# Plot final segmentation result + label marker
plt.subplot(1, 3, 3)
plt.imshow(segments_2d, cmap='gray')
plt.title("Segment Assignment + Seed Marker")
plt.colorbar()
# Draw seed marker
for (row, col) in label_coords:
    plt.scatter(col, row, s=200, facecolors='none', edgecolors='red', linewidths=2)

plt.tight_layout()
plt.show()

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: %.2f seconds" % elapsed_time)
