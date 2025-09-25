using LinearAlgebra
using Statistics
using DelimitedFiles

# Problem setup: 1×8 black/white image
nrows, ncols = 1, 8
N = nrows*ncols
bw = vcat(zeros(Float64, 4), ones(Float64, 4))  # 0..1
img = reshape(bw, nrows, ncols)
pix = vec(img) .* 255.0
σ = std(pix; corrected=false)

# Couplings J_x per horizontal bond: J = 2 - (4/σ)*|ΔI|
Jx = zeros(N-1)
for c in 1:(N-1)
    Δ = abs(pix[c] - pix[c+1])
    Jx[c] = 2.0 - (4.0/σ)*Δ
end

# Optional seeds: list of (site, label, strength) for local Sx fields
seeds = Tuple{Int,Float64,Float64}[]  # e.g., push!(seeds, (2, -1.0, 2.0))

# Single-site spin-1/2 operators (Sz basis)
σx = [0 1; 1 0];  σy = [0 -im; im 0];  σz = [1 0; 0 -1]
Sx = 0.5*σx;      Sy = 0.5*σy;          Sz = 0.5*σz
I2 = Matrix{ComplexF64}(I, 2, 2)

# Place local and two-site operators into the full 2^N space
function op_on(i::Int, A::AbstractMatrix{<:Complex}, N::Int)
    T = (i==1 ? A : I2)
    for s in 2:N
        T = kron(T, s==i ? A : I2)
    end
    return T
end

function op2_on(i::Int, j::Int, A::AbstractMatrix{<:Complex}, B::AbstractMatrix{<:Complex}, N::Int)
    @assert 1 <= i < j <= N
    T = (i==1 ? A : I2)
    for s in 2:N
        if s == j
            T = kron(T, B)
        elseif s == i
            # already placed A
        else
            T = kron(T, I2)
        end
    end
    return T
end

# Build dense Hamiltonian: H = sum_bonds (-SySy - SzSz - Jx*SxSx) + sum_seeds (-h*label*Sx)
dim = 2^N
H = zeros(ComplexF64, dim, dim)

for c in 1:(N-1)
    H .+= -op2_on(c, c+1, Sy, Sy, N)
    H .+= -op2_on(c, c+1, Sz, Sz, N)
    H .+= -Jx[c] * op2_on(c, c+1, Sx, Sx, N)
end

for (site, label, strength) in seeds
    H .+= -(strength*label) * op_on(site, Sx, N)
end

# Exact diagonalization
F = eigen(Hermitian(H))
evals = F.values
V = F.vectors  # columns are eigenvectors |E_α>

# Initial states in Sz basis (dense vectors)

# Uniform over all configurations: |+x>^⊗N
function plusx_vec(N::Int)
    inv√2 = inv(sqrt(2.0))
    v1 = ComplexF64[inv√2, inv√2]
    v = v1
    for _ in 2:N
        v = kron(v, v1)
    end
    v ./= norm(v)
    return v
end

# Equal superposition of all basis states with exactly M up-spins
function fixed_upspin_vec(N::Int, M::Int)
    v = zeros(ComplexF64, 2^N)
    for b in 0:(UInt(1)<<N)-1
        if count_ones(b) == M
            idx = Int(b) + 1  # Sz basis with site 1 as least-significant bit, |↑> mapped to bit=1
            v[idx] = 1
        end
    end
    if norm(v) > 0
        v ./= norm(v)
    end
    return v
end

v_plusx = plusx_vec(N)
M_up = round(Int, N * mean(img))
v_fixed = fixed_upspin_vec(N, M_up)

# Expansion coefficients |c_α|^2 for both initial states
coeffs_plusx = abs2.(conj(V)' * v_plusx)
coeffs_fixed = abs2.(conj(V)' * v_fixed)

println("N = $N, mean(image) = $(mean(img)), M_up = $M_up")
println("sum |c_α|^2 (plusx) = ", sum(coeffs_plusx))
println("sum |c_α|^2 (fixedM) = ", sum(coeffs_fixed))

# Save results
mkpath("examples")
writedlm("examples/ed_evals.csv", evals, ',')
writedlm("examples/ed_coeffs_plusx.csv", coeffs_plusx, ',')
writedlm("examples/ed_coeffs_fixedM$(M_up).csv", coeffs_fixed, ',')

# Print top contributors
perm_plusx = sortperm(coeffs_plusx; rev=true)
perm_fixed = sortperm(coeffs_fixed; rev=true)
println("top 5 (plusx): ", [(perm_plusx[k], coeffs_plusx[perm_plusx[k]]) for k in 1:min(5, length(perm_plusx))])
println("top 5 (fixedM): ", [(perm_fixed[k], coeffs_fixed[perm_fixed[k]]) for k in 1:min(5, length(perm_fixed))])
