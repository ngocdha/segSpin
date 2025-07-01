using DelimitedFiles
using Plots

gr()

# === Load data from CSV files ===
sx_matrix = readdlm("examples/sx_matrix.csv", ',')
img_reshaped = readdlm("examples/img_reshaped.csv", ',')
segments = readdlm("examples/segments.csv", ',')

nrows, ncols = size(sx_matrix)

# === Circle marker for the seed site ===
center_x, center_y = 2, 1
radius = 0.2
theta = LinRange(0, 2π, 100)
x = center_x .+ radius * cos.(theta)
y = center_y .+ radius * sin.(theta)

# === Plot 1: Original grayscale image ===
p1 = heatmap(img_reshaped, c=:coolwarm, title="Input Image (16×16)", aspect_ratio=1, axis=false)
vline!(0.5:(ncols+0.5), c=:black, legend=false)
hline!(0.5:(nrows+0.5), c=:black)
plot!(x, y, linewidth=2, linecolor=:black)

# === Plot 2: ⟨Sx⟩ values ===
p2 = heatmap(sx_matrix, c=:coolwarm, title="⟨Sx⟩ Values", aspect_ratio=1, axis=false)
vline!(0.5:(ncols+0.5), c=:black, legend=false)
hline!(0.5:(nrows+0.5), c=:black)
plot!(x, y, linewidth=2, linecolor=:black)

# === Plot 3: Segmentation result ===
p3 = heatmap(segments, c=:coolwarm, title="Segmentation", aspect_ratio=1, axis=false)
vline!(0.5:(ncols+0.5), c=:black, legend=false)
hline!(0.5:(nrows+0.5), c=:black)
plot!(x, y, linewidth=2, linecolor=:black)

# === Combine plots and save ===
plot_combined = plot(p1, p2, p3, layout=(1, 3), size=(1200, 400), title="Quantum Segmentation Result on 16×16 Image")

savefig(plot_combined, "plots/quantum_segmentation_results_16x16.pdf")
println("Saved plot to plots/quantum_segmentation_results_16x16.pdf")
