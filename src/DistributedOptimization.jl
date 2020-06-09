module DistributedOptimization

using LinearAlgebra
using LightGraphs
using Plots
using Distributions 
using SparseArrays
using Statistics
using Colors

# Utility Functions
include("./utils/graph_utils.jl")
include("./utils/linalg_utils.jl")

# Problem Definitions
include("./problems/target_loc.jl")
include("./problems/sep_quad.jl")

# Plotting Functions
include("./utils/plotting.jl")

# Distributed Optimization Algorithms
include("./algos/push_sum.jl")
include("./algos/cadmm.jl")
include("./algos/dda.jl")
include("./algos/extra.jl")
include("./algos/dig.jl")
include("./algos/next.jl")

# Results
include("./results/nonconvex.jl")

include("./results/comparison.jl")
export
    convergence_check,
    xhist_comparison,
    topology_comparison

end # module
