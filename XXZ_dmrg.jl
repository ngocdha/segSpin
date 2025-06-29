using ITensors, ITensorMPS
using Plots

gr()

let
    # 1. Input image (1D)
    n = 48
    img = vcat(fill(255.0, 4), fill(1.0, 4), fill(255.0, 4), fill(1.0, 4), fill(255.0, 4), fill(1.0, 4),fill(255.0, 4), fill(1.0, 4), fill(255.0, 4), fill(1.0, 4), fill(255.0, 4), fill(1.0, 4))

    # 2. Compute similarity-based couplings
    J = zeros(n)
    for i in 1:(n - 1)
        diff = abs(img[i] - img[i + 1])
        J[i] = 2 - (4 / 256) * diff
    end

    # 3. Build ITensor spin chain
    sites = siteinds("S=1/2", n)
    opsum = OpSum()

    # (a) Add interaction terms
    for i in 1:(n - 1)
        opsum += -1, "Sy", i, "Sy", i + 1
        opsum += -1, "Sz", i, "Sz", i + 1
        opsum += -J[i], "Sx", i, "Sx", i + 1
    end

    # (b) Add label constraint at site 2 (+1 label)
    label_strength = 1.0
    label_site = 2
    opsum += -label_strength, "Sx", label_site

    # 4. Convert to MPO
    H = MPO(opsum, sites)

    # 5. Initial MPS & Run DMRG
    psi0 = randomMPS(sites, linkdims=10)
    sweeps = Sweeps(10)
    maxdim!(sweeps, 10, 20, 100, 200)
    cutoff!(sweeps, 1e-10)

    println("Running DMRG...")
    energy, psi = dmrg(H, psi0, sweeps)
    println("Ground state energy: ", energy)

    # 6. Measure ⟨Sx⟩
    sx_vals = [real(expect(psi, "Sx"; sites=i)) for i in 1:n]
    segments = map(x -> x >= 0 ? 1 : 0, sx_vals)

    # 7. Plot results
    plot(layout=(3, 1), size=(600, 800))

    bar!(1:n, img, label="Pixel Intensity", xlabel="Site", ylabel="Intensity", subplot=1)
    bar!(1:n, sx_vals, label="⟨Sx⟩", xlabel="Site", ylabel="⟨Sx⟩", color=:cyan, subplot=2)
    bar!(1:n, segments, label="Segment", xlabel="Site", ylabel="Class", ylim=(-0.5, 1.5), color=:orange, subplot=3)

    savefig("plots/xxz_segmentation_plot.pdf")
    println("Segmentation plot saved to xxz_segmentation_plot.pdf")
end
