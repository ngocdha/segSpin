using ITensors, ITensorMPS
using Statistics
using Plots
using MLDatasets
using Images, ImageTransformations
using NPZ   # <-- NEW: enables saving arrays
gr()

# ---------------- 2D geometry and couplings ----------------

function image_to_edges2d(img::AbstractMatrix{<:Real}; kappa::Float64=4.0)
    nrows, ncols = size(img)
    grid = Float64.(img)
    σ = std(vec(grid); corrected=false)
    σ = (σ > 0 ? σ : 1.0)

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

# H_I = -∑ ( Sx_i Sx_j + Sy_i Sy_j + Jz_ij Sz_i Sz_j )
function build_HI(sites, edges)
    opsI = OpSum()
    for (i,j,Jz) in edges
        opsI += -1.0, "Sx", i, "Sx", j
        opsI += -1.0, "Sy", i, "Sy", j
        opsI += -Jz,  "Sz", i, "Sz", j
    end
    return MPO(opsI, sites)
end

# ---------------- low-energy states + truncated Gibbs ----------------

function lowest_k_states(H::MPO, sites; k::Int=3, sweeps::Sweeps=Sweeps(8))
    psis = Vector{MPS}(undef, k)
    Es   = Vector{Float64}(undef, k)

    E1, ψ1 = dmrg(H, randomMPS(sites, linkdims=20), sweeps)
    psis[1] = ψ1
    Es[1] = E1

    for m in 2:k
        ψ0 = randomMPS(sites, linkdims=20)
        Em, ψm = dmrg(H, psis[1:m-1], ψ0, sweeps)
        psis[m] = ψm
        Es[m] = Em
    end

    return Es, psis
end

weights(E, β) = (w = exp.(-β .* E); w ./= sum(w))

# ⟨Sz_ℓ Sz_i⟩ from truncated Gibbs using (E,ψ) of H_I
function trunc_SzℓSzi(sites, E, psis, ℓ; β=2.0)
    nt = length(sites)
    w = weights(E, β)
    out = zeros(nt)
    for i in 1:nt
        ops = OpSum()
        ops += 1.0, "Sz", ℓ, "Sz", i
        O = MPO(ops, sites)
        for (m,ψ) in enumerate(psis)
            out[i] += w[m] * real(inner(ψ, Apply(O, ψ)))
        end
    end
    return out
end

# ---------------- main wrapper ----------------

function truncated_corr_dmrg_2d(img::AbstractMatrix{<:Real};
        beta::Float64=2.0,
        kappa::Float64=4.0,
        k::Int=2,
        label_rc::Tuple{Int,Int}=(1,1),
        maxdims=(40,150,400,800))

    geom = image_to_edges2d(img; kappa=kappa)
    sites = siteinds("S=1/2", geom.nt; conserve_qns=false)
    H_I = build_HI(sites, geom.edges)

    sweeps = Sweeps(10)
    maxdim!(sweeps, maxdims...)
    cutoff!(sweeps, 1e-10)

    E_I, ps_I = lowest_k_states(H_I, sites; k=k, sweeps=sweeps)

    ℓ_site = (label_rc[1]-1)*geom.ncols + label_rc[2]
    szl_szi_vec = trunc_SzℓSzi(sites, E_I, ps_I, ℓ_site; β=beta)

    szl_szi = reshape(szl_szi_vec, geom.nrows, geom.ncols)

    return (; szl_szi, nrows=geom.nrows, ncols=geom.ncols, label_site=ℓ_site)
end

# ---------------- MNIST example w/ saving arrays ----------------

let
    ds = MNIST(split = :train)
    imgs   = ds.features
    labels = ds.targets

    target_digit = 7
    idx = findfirst(==(target_digit), labels)
    idx === nothing && (idx = 1)

    img28_raw = imgs[:, :, idx]
    img28 = Float64.(img28_raw)

    scale = maximum(img28) ≤ 1 ? 255.0 : 1.0
    img28 .*= scale

    nr = 12
    img_small = imresize(img28, (nr, nr))

    betas = [0.1, 0.5, 1.0, 2.0, 5.0]
    label_rc = (8, 8)
    k = 5

    for β in betas
        println("Running β = $β")
        start = time()

        out = truncated_corr_dmrg_2d(
            img_small;
            beta    = β,
            kappa   = 2.0,
            k       = k,
            label_rc = label_rc,
            maxdims = (40,150,400,800)
        )

        runtime = time() - start
        println("DMRG runtime (β = $β) = $runtime seconds")

        # orientation fixes
        input_plot = reverse(permutedims(img_small), dims=1)
        corr_plot  = reverse(out.szl_szi, dims=1)

        # ----- NEW: Save numeric arrays -----
        npzwrite("saved/input_raw_$(labels[idx])_$(nr)x$(nr)_beta$(β).npy", img_small)
        npzwrite("saved/corr_raw_$(labels[idx])_$(nr)x$(nr)_beta$(β).npy", out.szl_szi)

        npzwrite("saved/input_plot_$(labels[idx])_$(nr)x$(nr)_beta$(β).npy", input_plot)
        npzwrite("saved/corr_plot_$(labels[idx])_$(nr)x$(nr)_beta$(β).npy", corr_plot)

        p0 = heatmap(input_plot,
                     aspect_ratio=1,
                     title="MNIST digit $(labels[idx]) ($(nr)×$(nr))",
                     colorbar=true,
                     axis=false)

        p1 = heatmap(2 .* corr_plot,
                     aspect_ratio=1,
                     title="Z-Spin Correlation (β = $β)",
                     colorbar=true,
                     axis=false)

        plt = plot(p0, p1, layout=(1,2), size=(1200,500))
        savefig(plt, "plots/mnist_$(labels[idx])_$(nr)x$(nr)_beta$(β).png")
    end
end
