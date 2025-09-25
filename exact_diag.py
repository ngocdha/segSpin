import numpy as np
from numpy.linalg import eigh
from functools import reduce

# Problem setup: 1×8 black/white "image"
n = 8
img01 = np.concatenate([np.zeros(4), np.ones(4)])  # first 4 black, last 4 white
pix = img01 * 255.0
sigma = pix.std(ddof=0)

# Couplings J_x
Jx = np.zeros(n - 1)
for i in range(n - 1):
    d = abs(pix[i] - pix[i + 1])
    Jx[i] = 2.0 - (4.0 / sigma) * d

# Seeds (empty by default)
seeds = []

# Single-site operators
sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sy = np.array([[0, 1j], [-1j, 0]], dtype=np.complex128)
sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
Sx = 0.5 * sx
Sy = 0.5 * sy
Sz = 0.5 * sz
I2 = np.eye(2, dtype=np.complex128)

# Helpers to embed operators
def kron_all(mats):
    return reduce(np.kron, mats)

def op_on(site, A):
    mats = [I2] * n
    mats[site] = A
    return kron_all(mats)

def op2_on(i, j, A, B):
    mats = [I2] * n
    mats[i] = A
    mats[j] = B
    return kron_all(mats)

# Build Hamiltonian
dim = 2 ** n
H = np.zeros((dim, dim), dtype=np.complex128)

for i in range(n - 1):
    H -= op2_on(i, i + 1, Sy, Sy)
    H -= op2_on(i, i + 1, Sz, Sz)
    H -= Jx[i] * op2_on(i, i + 1, Sx, Sx)

for (site, label, strength) in seeds:
    H -= (strength * label) * op_on(site, Sx)

# Diagonalize
evals, V = eigh(H)

# Initial states
def plusx_vec(n):
    v1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex128)
    v = v1
    for _ in range(1, n):
        v = np.kron(v, v1)
    v /= np.linalg.norm(v)
    return v

def fixed_upspin_vec(n, M):
    v = np.zeros(2**n, dtype=np.complex128)
    for b in range(2**n):
        if bin(b).count("1") == M:
            v[b] = 1.0
    if np.linalg.norm(v) > 0:
        v /= np.linalg.norm(v)
    return v

v_plusx = plusx_vec(n)
M_up = int(round(n * img01.mean()))
v_fixed = fixed_upspin_vec(n, M_up)

# Expansion coefficients
c2_plusx = np.abs(V.conj().T @ v_plusx) ** 2
c2_fixed = np.abs(V.conj().T @ v_fixed) ** 2

print(f"N = {n}, mean(image) = {img01.mean():.3f}, M_up = {M_up}")
print("sum |c_alpha|^2 (plusx) =", float(c2_plusx.sum()))
print("sum |c_alpha|^2 (fixedM) =", float(c2_fixed.sum()))

# Top contributors
topk = 5
idx_plusx = np.argsort(-c2_plusx)[:topk]
idx_fixed = np.argsort(-c2_fixed)[:topk]
print("top 5 (plusx):", [(int(i), float(c2_plusx[i])) for i in idx_plusx])
print("top 5 (fixedM):", [(int(i), float(c2_fixed[i])) for i in idx_fixed])

print("V(70) = ", V[70])


# verify if state with highest c_alpha corresponds with ground state