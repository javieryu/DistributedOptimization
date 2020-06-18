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
              MAX_CYCLES::Int64=500, rho::Float64=0.5, recordx::Bool=true)

    # Convenience variables
    graph = prob.graph # communication graph
    n = prob.dim_opt # optimization var dimension
    N = prob.N # number of nodes

    # Initialize node ADMM structs
    nodes = [CADMMNode(n, x_inits[i], i) for i in 1:N]
    neighs = [neighbors(graph, i) for i in 1:N]

    # Initialize variables to record history
    if recordx
        xhist = Dict{Int, Array{Float64, 2}}()
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

    # For recording communication overhead
    comm_total = 0
    comm_hist = zeros(MAX_CYCLES, 1)
    comm_unit = sizeof(nodes[1].x) # so we don't need to call sizeof 25,000 times

    # Optimization main loop
    for t in 2:MAX_CYCLES
        # Local node updates
        for node in nodes
            # Communication Phase
            xj_sum = zeros(n, 1) # Sum of neighbors' x_prev
            for neigh in neighs[node.id]
                xj_sum += nodes[neigh].x_prev
                comm_total += comm_unit
            end

            # P-Update
            di = node.deg   
            node.p += rho .* (di .* node.x_prev - xj_sum)

            # X-Update
            rhs = rho .* (di .* node.x_prev + xj_sum) - node.p - prob.b[node.id]
            node.x = node.Jinv * rhs
        end

        # Update node x_prevs and record relevant information
        for node in nodes
            if recordx
                xhist[node.id][:, t] = node.x
            end

            node.x_prev = node.x
        end

        # Record Communication overhead
        comm_hist[t] = comm_total
    end
    
    # Compose array of final values
    fvals = Array{Float64}(undef, (n, N))
    for node in nodes
        fvals[:, node.id] = node.x
    end

    return fvals, xhist, comm_hist
end
