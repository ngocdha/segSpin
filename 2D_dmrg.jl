using ITensors, ITensorMPS
using Images, ImageIO, ImageTransformations, Interpolations
using Statistics
using Plots

gr()

let
    # 1. Load and process grayscale image (expected to be 8x8 or resizable to 8x8)
    img = load("mario_gray.png")
    img_array = Float64.(channelview(img))
    img_resized = imresize(img_array, (8, 8), interp=BSpline(Linear()))
    img_flat = vec(img_resized) .* 255
    nrows, ncols = size(img_resized)
    nt = nrows * ncols

    # 2. Compute similarity-based couplings (horizontal & vertical)
    Jh = zeros(nt)
    Jv = zeros(nt)

    for i in 1:nt
        if i % ncols != 0
            diff = abs(img_flat[i] - img_flat[i + 1])
            Jh[i] = 2 - (4 / 256) * diff
        end
        if i <= nt - ncols
            diff = abs(img_flat[i] - img_flat[i + ncols])
            Jv[i] = 2 - (4 / 256) * diff
        end
    end

    # 3. Hamiltonian construction
    sites = siteinds("S=1/2", nt)
    opsum = OpSum()

    for i in 1:nt
        if i % ncols != 0
            opsum += -1, "Sy", i, "Sy", i + 1
            opsum += -1, "Sz", i, "Sz", i + 1
            opsum += -Jh[i], "Sx", i, "Sx", i + 1
        end
        if i <= nt - ncols
            opsum += -1, "Sy", i, "Sy", i + ncols
            opsum += -1, "Sz", i, "Sz", i + ncols
            opsum += -Jv[i], "Sx", i, "Sx", i + ncols
        end
    end

    # Label constraint
    opsum += -1.0, "Sx", 2

    H = MPO(opsum, sites)

    # 4. DMRG
    psi0 = randomMPS(sites, linkdims=10)
    sweeps = Sweeps(10)
    maxdim!(sweeps, 10, 20, 100, 200)
    cutoff!(sweeps, 1e-10)

    println("Running DMRG...")
    energy, psi = dmrg(H, psi0, sweeps)
    println("Ground state energy: ", energy)

    # 5. Expectation values
    sx_vals = [real(expect(psi, "Sx"; sites=i)) for i in 1:nt]
    segments = reshape(map(x -> x ≥ 0 ? 1.0 : 0.0, sx_vals), nrows, ncols)

    # 6. Plot
    img_reshaped = reshape(img_flat, nrows, ncols)
    sx_matrix = reshape(sx_vals, nrows, ncols)

    plot(
        plot(img_resized, c=:gray, title="Grayscale Image", aspect_ratio=1, axis=false, colorbar=false),
        heatmap(sx_matrix, c=:coolwarm, title="⟨Sx⟩ Values", aspect_ratio=1, axis=false),
        plot(segments, c=:gray, title="Segmentation", aspect_ratio=1, axis=false),
        layout=(1, 3),
        size=(1200, 400)
    )

    savefig("plots/quantum_segmentation_results_2D.pdf")
    println("Saved plot to quantum_segmentation_results_2D.pdf")
end
