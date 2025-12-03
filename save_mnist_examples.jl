using MLDatasets
using Images, FileIO, ImageCore

mkpath("mnist_examples")

ds = MNIST(split = :train)
imgs   = ds.features
labels = ds.targets

for digit in 0:9
    idx = findfirst(==(digit), labels)

    # Raw MNIST image is typically UInt8 0–255 or 0–1; convert to Float64
    img_raw = Float64.(imgs[:, :, idx])

    # Normalize to [0,1]
    maxval = maximum(img_raw)
    img_norm = maxval == 0 ? img_raw : img_raw ./ maxval

    # Convert to grayscale image type
    img_gray = Gray.(img_norm)

    img_cw = reverse(permutedims(img_gray, (2,1)), dims=1)
    img_fixed = reverse(img_cw, dims=1)


    # Save as PNG
    save("mnist_examples/$(digit).png", img_fixed)
end

println("Saved MNIST images to mnist_examples/")
