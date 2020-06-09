mutable struct SqDistTargetLoc
    # A 2-D target location problem with a square distance measurement model
    # Following problem definition in:
    # "NEXT: In-Network Non-convex Optimization" Paolo Di Lorenzo et al
    #
    # Problem is bounded to the unitary square

    sp::Array{Float64, 2} # dim x N Sensor positions
    grnd_tru::Array{Float64, 2} # dim x T Ground truth target locations
    meas::Array{Float64, 2} # N x T matrix of measurements

    N::Int64 # Number of agents
    T::Int64 # Number of Targets
    dim_opt::Int64 # Dimension of problem in right now only 2, TODO 3D case?
    graph::SimpleGraph # Communication Graph

    function SqDistTargetLoc(T::Int64, pts::Array{Float64, 2},
                             var::Float64, graph::SimpleGraph;
                             targ_poses::Union{Array{Float64, 2}, Nothing}=nothing)
        # Inputs:
        # T - Number of targets
        # pts - matrix of sensor positions from LightGraphs.euclidean graph
        # var - sensor noise variance
        # graph - sensor communication graph
        
        prob = new() 
        prob.sp = pts
        prob.N = nv(graph)
        prob.T = T
        prob.graph = graph
        prob.dim_opt = 2

        if targ_poses == nothing
            prob.grnd_tru = rand(prob.dim_opt, T)
        else
            if size(targ_poses)[2] != T
                println("Specified target number and target positions don't match.")
                prob.T = size(targ_poses)[2]
            end
            prob.grnd_tru = targ_poses
        end

        prob.meas = zeros(prob.N, prob.T)

        # Generate measurements
        for i in 1:prob.N
            for targ in 1:prob.T
                prob.meas[i, targ] = norm(prob.sp[:, i] - prob.grnd_tru[:, targ])^2
            end
        end

        # Create and add noise to each measurement
        noise = rand(Normal(0.0, var), prob.N, T)
        prob.meas += noise

        return prob
    end
end

function local_gradient(prob::SqDistTargetLoc, id::Int64, x::Array{Float64, 2})
    # Returns gradient at node i of shape (x_i1, x_i2, ..., x_iT)
    # To reshape to vector: reshape(grad, :, 1)
    grad = zeros(prob.dim_opt, prob.T)

    w = prob.sp[:, id]
    for t in 1:prob.T
        cnst = -4.0 * (prob.meas[id, t] - norm(x[:, t] - w)^2) 
        grad[:, t] = cnst .* (x[:, t] - w)
    end

    return grad
end

function global_gradient(prob::SqDistTargetLoc, x::Array{Float64, 2})
    grad = zeros(prob.dim_opt, prob.T)
    for n in 1:prob.N
        grad += local_gradient(prob, n, x)
    end
    return grad
end

function proj2feas(prob::SqDistTargetLoc, x::Array{Float64, 2})
    # returns the projection of x into the [0, 1] x [0, 1] space 
    projx = zeros(size(x)) 
    for i in 1:length(x)
        projx[i] = min(max(x[i], 0.0), 1.0) 
    end
    return projx
end

function random_feasible_inits(prob::SqDistTargetLoc)
    x_inits = Dict{Int64, Array{Float64, 2}}()

    for i in 1:prob.N
        x_inits[i] = rand(prob.dim_opt, prob.T)
    end

    return x_inits
end

function compute_stationarity_series(prob::SqDistTargetLoc,
                                     xhist::Dict{Int64, Array{Float64, 3}})
    num_its = size(xhist[1])[3]
    stat = zeros(num_its, 1) 

    for time in 1:num_its
        # Compute mean x
        xmean = zeros(prob.dim_opt, prob.T)
        for n in 1:prob.N
            xmean += xhist[n][:, :, time] 
        end
        xmean /= prob.N

        # Compute global gradient at xmean
        grad = global_gradient(prob, xmean)
        #display(grad)
        #display(proj2feas(prob, xmean - grad))
        # Compute stationarity
        stat[time] = norm(reshape(xmean, :) - reshape(proj2feas(prob, xmean - grad), :), Inf)
    end

    return stat
end
