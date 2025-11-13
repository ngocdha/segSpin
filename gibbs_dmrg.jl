using ITensors, ITensorMPS
using Statistics
using Plots

# image -> edges with Jz = 2 - (kappa/sigma)*|I - I'|
function image_to_edges(img; kappa::Float64=4.0)
    if ndims(img) == 1
        nrows, ncols = 1, length(img)
        grid = reshape(Float64.(img), 1, :)
    else
        nrows, ncols = size(img)
        grid = Float64.(img)
    end
    σ = std(vec(grid); corrected=false); σ = (σ > 0 ? σ : 1.0)
    edges = Tuple{Int,Int,Float64}[]
    for r in 1:nrows, c in 1:ncols
        i = (r-1)*ncols + c
        if c+1 <= ncols
            diff = abs(grid[r,c] - grid[r,c+1])
            push!(edges, (i, i+1, 2.0 - (kappa/σ)*diff))
        end
        if r+1 <= nrows
            diff = abs(grid[r,c] - grid[r+1,c])
            j = r*ncols + c
            push!(edges, (i, j, 2.0 - (kappa/σ)*diff))
        end
    end
    nt = nrows*ncols
    return (nrows=nrows, ncols=ncols, nt=nt, edges=edges)
end

# Build H_I (Sz-Sz) and H_l (Sz labels) as MPOs
function build_HI_Hl(sites, edges; seeds=Tuple{Int,Float64,Float64}[])
    opsI = OpSum()
    for (i,j,Jz) in edges
        opsI += -Jz, "Sz", i, "Sz", j
    end
    H_I = MPO(opsI, sites)

    opsL = OpSum()
    for (site,label,strength) in seeds
        opsL += -(strength*label), "Sz", site
    end
    H_l = MPO(opsL, sites)
    return H_I, H_l
end

# Lowest k states by excited-state DMRG
function lowest_k_states(H::MPO, sites; k::Int=4, sweeps::Sweeps=Sweeps(8))
    psis = Vector{MPS}(undef, k)
    Es   = Vector{Float64}(undef, k)
    E1, ψ1 = dmrg(H, randomMPS(sites, linkdims=20), sweeps)
    psis[1] = ψ1; Es[1] = E1
    for m in 2:k
        ψ0 = randomMPS(sites, linkdims=20)
        Em, ψm = dmrg(H, psis[1:m-1], ψ0, sweeps)
        psis[m] = ψm; Es[m] = Em
    end
    return Es, psis
end

weights(E, β) = (w = exp.(-β .* E); w ./= sum(w))

# <Sz_i> from truncated Gibbs using (E,ψ) of H_full
function trunc_Sz_i(sites, E, psis; β=2.0)
    nt = length(sites); w = weights(E, β); out = zeros(nt)
    for i in 1:nt
        for (m,ψ) in enumerate(psis)
            out[i] += w[m]*expect(ψ, "Sz"; sites=i)
        end
    end
    out
end

# <Sz_ℓ Sz_i> from truncated Gibbs using (E,ψ) of H_I
function trunc_SzℓSzi(sites, E, psis, ℓ; β=2.0)
    nt = length(sites); w = weights(E, β); out = zeros(nt)
    for i in 1:nt
        ops = OpSum(); ops += 1.0, "Sz", ℓ, "Sz", i
        O = MPO(ops, sites)
        for (m,ψ) in enumerate(psis)
            out[i] += w[m]*real(inner(ψ, O, ψ))
        end
    end
    out
end

# Truncated-Gibbs driver (DMRG eigensolver approach)
function truncated_gibbs_dmrg(img; beta=2.0, mu=0.0, seeds=Tuple{Int,Float64,Float64}[],
                              kappa=4.0, k=4, label_site=1,
                              maxdims=(10,20,100,200))
    geom = image_to_edges(img; kappa=kappa)
    sites = siteinds("S=1/2", geom.nt; conserve_qns=false)
    H_I, H_l = build_HI_Hl(sites, geom.edges; seeds=seeds)
    H_full = H_I + mu*H_l

    sweeps = Sweeps(10); maxdim!(sweeps, maxdims...); cutoff!(sweeps, 1e-10)

    E_full, ps_full = lowest_k_states(H_full, sites; k=k, sweeps=sweeps)
    sz_i = trunc_Sz_i(sites, E_full, ps_full; β=beta)

    E_I, ps_I = lowest_k_states(H_I, sites; k=k, sweeps=sweeps)
    szlszi = trunc_SzℓSzi(sites, E_I, ps_I, label_site; β=beta)

    return (; sz_i, szlszi, n=geom.nt)
end

# ---- example plot like Python ----
gr()
let
    img1d = vcat(zeros(5), fill(255.0, 5))  # 1×10 toy
    beta = 2.0
    mu = 0.0
    label_site = 1
    seeds = [(label_site, +1.0, 5.0)]  # used only if mu ≠ 0
    k = 10

    out = truncated_gibbs_dmrg(img1d; beta=beta, mu=mu, seeds=seeds,
                               kappa=4.0, k=k, label_site=label_site)

    x = 1:out.n
    p = plot(x, out.sz_i, marker=:circle, label="⟨Sᶻ_i⟩",
             xlabel="Pixel index i", ylabel="Value",
             title="⟨Sᶻ_i⟩ and 2⟨Sᶻ_ℓ Sᶻ_i⟩ (truncated Gibbs via DMRG)")
    plot!(x, 2 .* out.szlszi, marker=:square, linestyle=:dash, label="2⟨Sᶻ_ℓ Sᶻ_i⟩")
    display(p)
end
