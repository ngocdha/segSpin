import numpy as np
from numpy.linalg import eigh, norm
from functools import reduce
from scipy.linalg import expm

# 3 qubits: 0 = ancilla, 1 = pixel 1 (black), 2 = pixel 2 (white)
n = 3

# single-site operators (Sz basis)
sx = np.array([[0,1],[1,0]], dtype=np.complex128)
sy = np.array([[0,1j],[-1j,0]], dtype=np.complex128)
sz = np.array([[1,0],[0,-1]], dtype=np.complex128)
Sx, Sy, Sz = 0.5*sx, 0.5*sy, 0.5*sz
I2 = np.eye(2, dtype=np.complex128)

def kron_all(mats):
    return reduce(np.kron, mats)

def op_on(site, A):
    mats = [I2]*n
    mats[site] = A
    return kron_all(mats)

def op2_on(i, j, A, B):
    mats = [I2]*n
    mats[i] = A
    mats[j] = B
    return kron_all(mats)

# two-pixel contrast → anisotropy on SzSz (bond 1–2)
pix = np.array([0.0, 255.0])
sigma = pix.std(ddof=0)
alpha, beta = 2.0, 4.0/sigma
Delta = alpha - beta*abs(pix[0]-pix[1])   # with these numbers: -6.0

# build H pieces
epsilon = 1e-1             # ancilla–pixel1 XY
seed_strength = 10.0       # seed on pixel 2 along z, label +1
H01    = epsilon * (op2_on(0,1,Sx,Sx) + op2_on(0,1,Sy,Sy))
H12_xy = -(0.5) * (op2_on(1,2,Sx,Sx) + op2_on(1,2,Sy,Sy))
H12_zz = -(0.5) * (Delta * op2_on(1,2,Sz,Sz))
Hz2    =  seed_strength * op_on(2, Sz)

H = H01 + H12_xy + H12_zz + Hz2

# initial state |010> + |001| (normalized)
ket0 = np.array([1,0], dtype=np.complex128)
ket1 = np.array([0,1], dtype=np.complex128)
psi_010 = kron_all([ket0, ket1, ket0])
psi_001 = kron_all([ket0, ket0, ket1])
psi = psi_010 + psi_001
psi /= norm(psi)

# time evolution parameters
T  = 80.0
Nt = 4000
dt = T / Nt

# precompute full-step unitary once: U = exp(-i H dt)
U = expm(-1j * H * dt)

# observables to record
Sz_ops = [op_on(j, Sz) for j in range(3)]
def expect(psi, O):
    return float(np.real(np.vdot(psi, O @ psi)))

ts = np.linspace(0.0, T, Nt+1)
sz1_t = np.empty(Nt+1)
sz2_t = np.empty(Nt+1)
sz1_t[0] = expect(psi, Sz_ops[1])
sz2_t[0] = expect(psi, Sz_ops[2])

# evolve by repeated application of U
for k in range(1, Nt+1):
    psi = U @ psi
    # optional renorm to control tiny numerical drift
    psi /= norm(psi)
    sz1_t[k] = expect(psi, Sz_ops[1])
    sz2_t[k] = expect(psi, Sz_ops[2])

print("Delta(1,2) =", float(Delta))
print("Avg <Sz(1)> over time:", float(sz1_t.mean()))
print("Avg <Sz(2)> over time:", float(sz2_t.mean()))

# diagonal-ensemble check (optional)
evals, V = eigh(H)
c = V.conj().T @ (psi_010 + psi_001) / norm(psi_010 + psi_001)
c2 = np.abs(c)**2
def diag_avg(O):
    Od = np.array([expect(V[:,a], O) for a in range(2**n)])
    return float(c2 @ Od)
print("Diag-ensemble <Sz(1)> =", diag_avg(Sz_ops[1]))
print("Diag-ensemble <Sz(2)> =", diag_avg(Sz_ops[2]))

# segmentation from long-time averages: O(i)=1/2+Sz(i)
Oi_bar = 0.5 + np.array([sz1_t.mean(), sz2_t.mean()])  # sites 1 and 2
segments = (Oi_bar >= 0.5).astype(int)
print("Ō (sites 1,2) =", Oi_bar, "segments =", segments)
