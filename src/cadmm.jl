mutable struct CADMMNode
    id::Int64
    deg::Int64
    x::Array{Float64, 2}
    x_prev::Array{Float64, 2}
    p::Array{Float64, 2}
    Jinv::Union{Array{Float64, 2}, Nothing}

    function CADMMNode(dim::Int, x_init::Array{Float64, 2}, id::Int)
        node = new()
        node.x = x_init
        node.x_prev = x_init
        node.p = zeros(dim, 1)
        node.deg = 0
        node.id = id
        node.Jinv = nothing
        
        return node
    end
end

function cadmm(prob::SeperableQuadratic, x_inits::Dict{Int, Array{Float64, 2}};
              x_cent::Union{Array{Float64, 2}, Nothing}=nothing,
              MAX_CYCLES::Int64=500, rho::Float64=0.5,
              recordx::Bool=false)

    # Convenience variables
    graph = prob.graph # communication graph
    n = prob.dim_opt # optimization var dimension
    N = prob.N # number of nodes

    # Initialize node ADMM structs
    nodes = [CADMMNode(n, x_inits[i], i) for i in 1:N]
    neighs = [neighbors(graph, i) for i in 1:N]

    # Initialize variables to analyze convergence
    errors = zeros(N, MAX_CYCLES - 1)
    xhist = Dict{Int, Array{Float64, 2}}()

    if recordx
        for i in 1:N
            xhist[i] = zeros(n, MAX_CYCLES)
            xhist[i][:, 1] = x_inits[i]
        end
    end

    # Local precomputations and node variable initializations
    # Fully distributed
    for i in 1:N
        nodes[i].deg = length(neighs[i])

        di = nodes[i].deg
        nodes[i].Jinv = inv(2.0 .* prob.A[i] + 2.0 * rho * di .* I(n))
    end

    # Optimization main loop
    for t in 2:MAX_CYCLES
        # Local node updates
        for node in nodes
            # Communication Phase
            xj_sum = zeros(n, 1) # Sum of neighbors' x_prev
            for neigh in neighs[node.id]
                xj_sum += nodes[neigh].x_prev
            end

            # P-Update
            di = node.deg   
            node.p += rho .* (di .* node.x_prev - xj_sum)

            # X-Update
            rhs = rho .* (di .* node.x_prev + xj_sum) - node.p - prob.b[node.id]
            node.x = node.Jinv * rhs

            # Store relevant variables for analysis
            if x_cent == nothing
                errors[node.id, t - 1] = norm(node.x_prev - node.x)
            else
                errors[node.id, t - 1] = norm(x_cent - node.x)
            end

            if recordx
                xhist[node.id][:, t] = node.x
            end
        end

        # Update node x_prevs
        for node in nodes
            node.x_prev = node.x
        end
    end

    return errors, xhist
end
