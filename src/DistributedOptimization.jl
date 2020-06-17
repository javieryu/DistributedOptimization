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
export
    generate_disk_graph,
    generate_random_graph

include("./utils/linalg_utils.jl")

include("./utils/error_metrics.jl")
export
    norm2_error

# Problem Definitions
include("./problems/target_loc.jl")
export
    SqDistTargetLoc,
    random_feasible_inits,
    compute_stationarity_series

include("./problems/sep_quad.jl")
export
    SeperableQuadratic,
    solve_centralized,
    solve_local,
    get_max_cond

# Plotting Functions
include("./utils/plotting.jl")
export
    plot_all_errors!,
    plot_bounded_errors!,
    plot_stat!,
    plot_mean_error!,
    plot_xhist!,
    plot_xcent!,
    plot_xinits!,
    plot_sq_dis_traj!,
    plot_cost_contour!,
    get_xhist_limits

# Distributed Optimization Algorithms
include("./algos/push_sum.jl")
include("./algos/cadmm.jl")
include("./algos/dda.jl")
include("./algos/extra.jl")
include("./algos/dig.jl")
include("./algos/next.jl")
export
    push_sum,
    cadmm,
    dda,
    extra,
    dig,
    next

end # module
