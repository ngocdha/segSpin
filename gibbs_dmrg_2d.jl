using ITensors, ITensorMPS
using Statistics
using Plots
gr()

# 2D image -> edges, Jz = 2 - (kappa/sigma)*|I - I'|, 4-neighbor
function image_to_edges2d(img::AbstractMatrix{<:Real}; kappa::Float64=4.0)
    nrows, ncols = size(img)
    grid = Float64.(img)
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
    return (nrows=nrows, ncols=ncols, nt=nrows*ncols, edges=edges)
end

# Build H_I (Sz-Sz) and H_l (local Sz fields) as MPOs
# seeds_site :: Vector{(site::Int, label::Float64, strength::Float64)} with 1-based site
function build_HI_Hl(sites, edges; seeds_site=Tuple{Int,Float64,Float64}[])
    opsI = OpSum()
    for (i,j,Jz) in edges
        opsI += -Jz, "Sz", i, "Sz", j
    end
    H_I = MPO(opsI, sites)

    opsL = OpSum()
    for (site,label,strength) in seeds_site
        opsL += -(strength*label), "Sz", site
    end
    H_l = MPO(opsL, sites)
    return H_I, H_l
end

# (r,c,label,strength) -> (site,label,strength)
convert_seeds_rc_to_site(seeds_rc, ncols) = [( (r-1)*ncols + c, lab, str ) for (r,c,lab,str) in seeds_rc]

# Lowest k eigenstates by excited-state DMRG
function lowest_k_states(H::MPO, sites; k::Int=3, sweeps::Sweeps=Sweeps(8))
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

# <Sz_ℓ Sz_i> from truncated Gibbs using (E,ψ) of H_I (uses Apply fix for inner)
function trunc_SzℓSzi(sites, E, psis, ℓ; β=2.0)
    nt = length(sites); w = weights(E, β); out = zeros(nt)
    for i in 1:nt
        ops = OpSum(); ops += 1.0, "Sz", ℓ, "Sz", i
        O = MPO(ops, sites)
        for (m,ψ) in enumerate(psis)
            out[i] += w[m] * real(inner(ψ, Apply(O, ψ)))
        end
    end
    out
end

# Truncated Gibbs via DMRG on 2D image
# seeds_rc :: Vector{(r,c,label,strength)} with label ∈ {±1}
function truncated_gibbs_dmrg_2d(img::AbstractMatrix{<:Real};
        beta::Float64=2.0, mu::Float64=0.0, seeds_rc=Tuple{Int,Int,Float64,Float64}[],
        kappa::Float64=4.0, k::Int=3, label_rc::Tuple{Int,Int}=(1,1),
        maxdims=(20,50,100,200))

    geom = image_to_edges2d(img; kappa=kappa)
    sites = siteinds("S=1/2", geom.nt; conserve_qns=false)

    seeds_site = convert_seeds_rc_to_site(seeds_rc, geom.ncols)
    H_I, H_l = build_HI_Hl(sites, geom.edges; seeds_site=seeds_site)
    H_full = H_I + mu*H_l

    sweeps = Sweeps(10); maxdim!(sweeps, maxdims...); cutoff!(sweeps, 1e-10)

    E_full, ps_full = lowest_k_states(H_full, sites; k=k, sweeps=sweeps)
    sz_i_vec = trunc_Sz_i(sites, E_full, ps_full; β=beta)

    E_I, ps_I = lowest_k_states(H_I, sites; k=k, sweeps=sweeps)
    ℓ_site = (label_rc[1]-1)*geom.ncols + label_rc[2]
    szl_szi_vec = trunc_SzℓSzi(sites, E_I, ps_I, ℓ_site; β=beta)

    sz_i    = reshape(sz_i_vec,    geom.nrows, geom.ncols)
    szl_szi = reshape(szl_szi_vec, geom.nrows, geom.ncols)

    return (; sz_i, szl_szi, nrows=geom.nrows, ncols=geom.ncols, label_site=ℓ_site)
end

using MLDatasets
using Images, ImageTransformations
using Plots
gr()

# --- MNIST → 16×16 → truncated Gibbs DMRG + correctly oriented plots ---

let
    # Load MNIST training set
    ds = MNIST(split = :train)
    imgs   = ds.features      # 28×28×N, typically UInt8
    labels = ds.targets       # Vector{Int}

    # Pick a digit and find first example of that digit
    target_digit = 8
    idx = findfirst(==(target_digit), labels)
    idx === nothing && (idx = 1)

    # Extract 28×28 image and convert to Float64
    img28_raw = imgs[:, :, idx]
    img28 = Float64.(img28_raw)

    # Scale to [0,255] if needed
    scale = maximum(img28) ≤ 1 ? 255.0 : 1.0
    img28 .*= scale

    # Downscale to 16×16
    img16 = imresize(img28, (16, 16))

    # Truncated Gibbs via DMRG on 16×16 MNIST digit
    beta = 2.0
    mu   = 1.0                 
    label_rc = (8, 8)          
    seeds_rc = [(8,8,+1.0,5.0)]  
    k = 5

    out = truncated_gibbs_dmrg_2d(
        img16;
        beta    = beta,
        mu      = mu,
        seeds_rc = seeds_rc,
        kappa   = 4.0,
        k       = k,
        label_rc = label_rc,
        maxdims = (50,150,400,800)
    )

    # --- Orientation fix: transpose + flip vertically for all 3 panels ---

    input_plot = reverse(permutedims(img16), dims=1)
    sz_plot    = reverse(permutedims(out.sz_i), dims=1)
    szlz_plot  = reverse(permutedims(out.szl_szi), dims=1)

    # --- Plots: input, ⟨Sᶻ⟩, 2⟨Sᶻ_ℓ Sᶻ⟩ ---

    p0 = heatmap(input_plot,
                 aspect_ratio=1,
                 title="MNIST digit $(labels[idx]) (16×16)",
                 colorbar=false,
                 axis=false)

    p1 = heatmap(sz_plot,
                 aspect_ratio=1,
                 title="⟨Sᶻ⟩  (H_I + μH_l)",
                 colorbar=true,
                 axis=false)

    p2 = heatmap(2 .* szlz_plot,
                 aspect_ratio=1,
                 title="2⟨Sᶻ_ℓ Sᶻ⟩  (H_I)",
                 colorbar=true,
                 axis=false)

    plot(p0, p1, p2, layout=(1,3), size=(1600,500))
end




