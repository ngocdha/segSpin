import numpy as np
from numpy.linalg import eigh
from functools import reduce

# qubit order: 0 = ancilla, 1 = pixel (black), 2 = pixel (white)
n = 3

# single-site spin-1/2 operators (Sz basis)
sx = np.array([[0, 1],[1, 0]], dtype=np.complex128)
sy = np.array([[0, 1j],[-1j, 0]], dtype=np.complex128)
sz = np.array([[1, 0],[0, -1]], dtype=np.complex128)
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

# pixel 1 = black (0), pixel 2 = white (255)
pix = np.array([0.0, 255.0])
sigma = pix.std(ddof=0)            # std([0,255]) = 127.5
alpha = 2.0
beta  = 4.0/sigma
Delta = alpha - beta*abs(pix[0]-pix[1])   # = 2 - (4/127.5)*255 = -6.0
print("Delta(1,2) =", Delta)

dim = 2**n
H = np.zeros((dim, dim), dtype=np.complex128)

# pixel-pixel bond (1,2): -1/2 * (SxSx + SySy + Δ SzSz)
H += -(0.5) * ( op2_on(1,2,Sx,Sx) + op2_on(1,2,Sy,Sy) + Delta*op2_on(1,2,Sz,Sz) )

# seed on Z at pixel 2 (site index 2)
seed_strength = 10.0
seed_label = +1.0     # +1 favors |0> (up) along z
H += seed_strength * seed_label * op_on(2, Sz)

# ancilla–pixel1 coupling: epsilon*(Sx0 Sx1 + Sy0 Sy1)
epsilon = 1e-1
H += epsilon * ( op2_on(0,1,Sx,Sx) + op2_on(0,1,Sy,Sy) )

evals, V = eigh(H)
ground_idx = int(np.argmin(evals))
print("Eigenvalues:", evals.real)
print("Ground state index:", ground_idx, "Energy:", float(evals[ground_idx].real))

# basis ordering is (q0 ⊗ q1 ⊗ q2), with |0> = [1,0], |1> = [0,1]
ket0 = np.array([1,0], dtype=np.complex128)
ket1 = np.array([0,1], dtype=np.complex128)

psi_010 = kron_all([ket0, ket1, ket0])   # |0>_0 ⊗ |1>_1 ⊗ |0>_2
psi_001 = kron_all([ket0, ket0, ket1])   # |0>_0 ⊗ |0>_1 ⊗ |1>_2
psi0 = (psi_010 + psi_001)
psi0 /= np.linalg.norm(psi0)

# overlaps |c_alpha|^2 with eigenstates
c2 = np.abs(V.conj().T @ psi0)**2
print("Sum |c_alpha|^2 =", float(c2.sum()))
print("Top overlaps (index, weight):", [(int(i), float(c2[i])) for i in np.argsort(-c2)[:5]])

# a few expectation values in the ground state
gs = V[:, ground_idx]
def expect(state, O):
    return np.real(np.vdot(state, O @ state))

print("<Sz(2)> in ground state:", expect(gs, op_on(2, Sz)))
print("<Sz(1)> in ground state:", expect(gs, op_on(1, Sz)))
print("<Sx(0)Sx(1)> in ground state:", expect(gs, op2_on(0,1,Sx,Sx)))
print("<Sx(1)Sx(2)> in ground state:", expect(gs, op2_on(1,2,Sx,Sx)))
