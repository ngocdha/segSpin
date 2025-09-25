using ITensors, ITensorMPS
using Images, ImageIO, ImageTransformations, Interpolations
using Statistics
using Plots
using DelimitedFiles

gr()

let
    # 1. Load and process original grayscale image (16x16)
    img = load("mario_gray.png")
    img_array = Float64.(channelview(img))
    @assert size(img_array) == (16, 16) "Expected 16x16 image, got $(size(img_array))"

    img_flat = reverse(vec(img_array)) .* 255
    nrows, ncols = size(img_array)
    nt = nrows * ncols
    sigma = std(img_flat; corrected=false)

    # 2. Compute similarity-based couplings (horizontal & vertical)
    Jh = zeros(nt)
    Jv = zeros(nt)

    for i in 1:nt
        if i % ncols != 0
            diff = abs(img_flat[i] - img_flat[i + 1])
            #Jh[i] = 2 - (8 / 256) * diff
            Jh[i] = 2 - (6 / sigma) * diff
        end
        if i <= nt - ncols
            diff = abs(img_flat[i] - img_flat[i + ncols])
            #Jv[i] = 2 - (8 / 256) * diff
            Jv[i] = 2 - (6 / sigma) * diff
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

    # 4. Label constraints (seeds)
    seeds = [(200,-1.0)]
    for (site, label) in seeds
        opsum += -100.0 * label, "Sx", site
    end

    H = MPO(opsum, sites)

    # 5. DMRG
    psi0 = randomMPS(sites, linkdims=20)
    sweeps = Sweeps(10)
    maxdim!(sweeps, 10, 20, 100, 200)
    cutoff!(sweeps, 1e-10)

    println("Running DMRG...")
    energy, psi = dmrg(H, psi0, sweeps)
    println("Ground state energy: ", energy)

    # 6. Expectation values and segmentation
    sx_vals = [real(expect(psi, "Sx"; sites=i)) for i in 1:nt]
    segments = reshape(map(x -> x ≥ 0 ? 1.0 : 0.0, sx_vals), nrows, ncols)

    # 7. Prepare data for plotting
    img_reshaped = reshape(img_flat, nrows, ncols)
    sx_matrix = reshape(sx_vals, nrows, ncols)

    # 8. Save data to CSV
    writedlm("examples/sx_matrix.csv", sx_matrix, ',')
    writedlm("examples/img_reshaped.csv", img_reshaped, ',')
    writedlm("examples/segments.csv", segments, ',')

    println("Exported sx_matrix, img_reshaped, and segments to 'examples/' folder.")

    # 9. Plot results with seed circles
    function seed_circle(site)
        y = Int(fld(site - 1, ncols)) + 1
        x = Int(mod(site - 1, ncols)) + 1
        radius = 0.2
        θ = LinRange(0, 2π, 100)
        x .+ radius * cos.(θ), y .+ radius * sin.(θ)
    end

    p1 = heatmap(img_reshaped, c=:coolwarm, title="Input Image (16×16)", aspect_ratio=1, axis=false)
    p2 = heatmap(sx_matrix, c=:coolwarm, title="⟨Sx⟩ Values", aspect_ratio=1, axis=false)
    p3 = heatmap(segments, c=:coolwarm, title="Segmentation", aspect_ratio=1, axis=false)

    for (site, _) in seeds
        x_circle, y_circle = seed_circle(site)
        for p in (p1, p2, p3)
            vline!(p, 0.5:(ncols+0.5), c=:black, legend=false)
            hline!(p, 0.5:(nrows+0.5), c=:black)
            plot!(p, x_circle, y_circle, linewidth=2, linecolor=:black)
        end
    end

    final_plot = plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
    savefig(final_plot, "plots/quantum_segmentation_results_16x16_face_6.pdf")
    println("Saved plot to plots/quantum_segmentation_results_16x16_face_6.pdf")
end
