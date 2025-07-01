using ITensors, ITensorMPS
using Images, ImageIO, ImageTransformations, Interpolations
using Statistics
using Plots
using DelimitedFiles  # for CSV export

gr()

let
    # 1. Load and process original grayscale image (16x16)
    img = load("mario_gray.png")
    img_array = Float64.(channelview(img))
    @assert size(img_array) == (16, 16) "Expected 16x16 image, got $(size(img_array))"

    img_flat = reverse(vec(img_array)) .* 255
    nrows, ncols = size(img_array)
    nt = nrows * ncols

    # 2. Compute similarity-based couplings (horizontal & vertical)
    Jh = zeros(nt)
    Jv = zeros(nt)
    sigma = std(img_flat)/sqrt(nt)

    for i in 1:nt
        if i % ncols != 0
            diff = abs(img_flat[i] - img_flat[i + 1])
            Jh[i] = 2 - (2 / sigma) * diff
        end
        if i <= nt - ncols
            diff = abs(img_flat[i] - img_flat[i + ncols])
            Jv[i] = 2 - (2 / sigma) * diff
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
    opsum += -100.0, "Sx", 2

    H = MPO(opsum, sites)

    # 4. DMRG
    psi0 = randomMPS(sites, linkdims=20)
    sweeps = Sweeps(10)
    maxdim!(sweeps, 10, 20, 100, 200)
    cutoff!(sweeps, 1e-10)

    println("Running DMRG...")
    energy, psi = dmrg(H, psi0, sweeps)
    println("Ground state energy: ", energy)

    # 5. Expectation values
    sx_vals = [real(expect(psi, "Sx"; sites=i)) for i in 1:nt]
    segments = reshape(map(x -> x ≥ 0 ? 1.0 : 0.0, sx_vals), nrows, ncols)

    # 6. Plot results
    img_reshaped = reshape(img_flat, nrows, ncols)
    sx_matrix = reshape(sx_vals, nrows, ncols)

    # 7. Save data to examples folder
    mkpath("examples")
    writedlm("examples/sx_matrix.csv", sx_matrix, ',')
    writedlm("examples/img_reshaped.csv", img_reshaped, ',')
    writedlm("examples/segments.csv", segments, ',')

    println("Exported sx_matrix, img_reshaped, and segments to 'examples/' folder.")
end
