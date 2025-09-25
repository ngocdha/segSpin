const SMALL = true  # true = run toy 1×8 example, false = full 16×16 Mario

using ITensors, ITensorMPS
using Images, ImageIO
using Statistics
using Plots, DelimitedFiles

gr()

# Build uniform |+> state
function uniform_plusx_mps(sites)
    A = [let s = sites[i]; t = ITensor(s);
             t[s=>1] = inv(sqrt(2.0)); t[s=>2] = inv(sqrt(2.0)); t
         end for i in 1:length(sites)]
    ψ = MPS(A)
    orthogonalize!(ψ, 1); normalize!(ψ)
    return ψ
end

let
    # Load data
    if SMALL
        nrows, ncols = 1, 8
        nt = nrows*ncols
        img_reshaped = reshape(collect(range(0, 255; length=nt)), nrows, ncols)
        img_flat = vec(img_reshaped)
        sigma = std(img_flat; corrected=false)
        seeds = [(2,-1.0)]
    else
        img = load("mario_gray.png")
        img_array = Float64.(channelview(img))
        @assert size(img_array) == (16, 16)
        nrows, ncols = size(img_array)
        nt = nrows*ncols
        img_flat = reverse(vec(img_array)) .* 255
        img_reshaped = reshape(img_flat, nrows, ncols)
        sigma = std(img_flat; corrected=false)
        seeds = [(200,-1.0)]
    end

    # Couplings
    Jh = zeros(nt); Jv = zeros(nt)
    for i in 1:nt
        if i % ncols != 0
            diff = abs(img_reshaped[div(i-1,ncols)+1, mod(i-1,ncols)+1] -
                       img_reshaped[div(i-1,ncols)+1, mod(i-1,ncols)+2])
            Jh[i] = 2 - (4/sigma)*diff
        end
        if i <= nt-ncols
            diff = abs(img_reshaped[div(i-1,ncols)+1, mod(i-1,ncols)+1] -
                       img_reshaped[div(i-1,ncols)+2, mod(i-1,ncols)+1])
            Jv[i] = 2 - (4/sigma)*diff
        end
    end

    # Hamiltonian
    sites = siteinds("S=1/2", nt; conserve_qns=false)
    opsum = OpSum()
    for i in 1:nt
        if i % ncols != 0
            opsum += -1,"Sy",i,"Sy",i+1
            opsum += -1,"Sz",i,"Sz",i+1
            opsum += -Jh[i],"Sx",i,"Sx",i+1
        end
        if i <= nt-ncols
            opsum += -1,"Sy",i,"Sy",i+ncols
            opsum += -1,"Sz",i,"Sz",i+ncols
            opsum += -Jv[i],"Sx",i,"Sx",i+ncols
        end
    end
    for (site,label) in seeds
        opsum += -100.0*label,"Sx",site
    end
    H = MPO(opsum, sites)

    # DMRG
    sweeps = Sweeps(10); maxdim!(sweeps,10,20,100,200); cutoff!(sweeps,1e-10)
    E1, gs = dmrg(H, randomMPS(sites,linkdims=20), sweeps)

    psi1 = randomMPS(sites, linkdims=20)
    E2, es1 = dmrg(H, [gs], psi1, sweeps)

    println("E1 = $E1, E2 = $E2")

    # Overlaps and observables
    IC = uniform_plusx_mps(sites)

    println("norm(gs)^2   = ", norm(gs)^2)
    println("norm(es1)^2  = ", norm(es1)^2)
    println("norm(IC)^2   = ", norm(IC)^2)
    println("⟨gs|IC⟩       = ", inner(gs, IC))
    println("⟨es1|IC⟩      = ", inner(es1, IC))

    c1 = abs2(inner(gs,  IC)) / abs2(inner(gs,  gs))
    c2 = abs2(inner(es1, IC)) / abs2(inner(es1, es1))

    println("c1 = $c1, c2 = $c2, c1+c2 = $(c1+c2)")

    O1 = reshape([0.5+expect(gs,"Sz";sites=i) for i in 1:nt], nrows,ncols)
    O2 = reshape([0.5+expect(es1,"Sz";sites=i) for i in 1:nt], nrows,ncols)
    Obar = c1*O1 .+ c2*O2

    writedlm("examples/O1.csv", O1, ',')
    writedlm("examples/O2.csv", O2, ',')
    writedlm("examples/Obar.csv", Obar, ',')

    # Visualization
    sx_vals = [real(expect(gs,"Sx";sites=i)) for i in 1:nt]
    sx_matrix = reshape(sx_vals,nrows,ncols)
    segments = reshape((x->x>=0 ? 1.0 : 0.0).(sx_vals), nrows,ncols)

    p1 = heatmap(img_reshaped, c=:coolwarm, title=SMALL ? "Toy 1×8" : "Input 16×16",
                 aspect_ratio=1, axis=false)
    p2 = heatmap(sx_matrix, c=:coolwarm, title="⟨Sx⟩ (GS)", aspect_ratio=1, axis=false)
    p3 = heatmap(segments, c=:coolwarm, title="Segmentation", aspect_ratio=1, axis=false)

    final_plot = plot(p1,p2,p3,layout=(1,3),size=(1200,400))
    savefig(final_plot, SMALL ? "plots/toy_overlap.pdf" : "plots/mario_overlap.pdf")
end
