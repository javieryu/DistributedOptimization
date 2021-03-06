mutable struct PushSumNode
    id::Int64
    deg::Int64
    x::Array{Float64, 2}
    x_prev::Array{Float64, 2}
    w::Array{Float64, 2}
    z::Array{Float64, 2}
    y::Float64
    y_prev::Float64

    function PushSumNode(dim::Int, x_init::Array{Float64, 2}, id::Int)
        node = new()
        node.x = x_init
        node.x_prev = x_init
        node.w = zeros(dim, 1)
        node.z = zeros(dim, 1)
        node.y = 1.0
        node.y_prev = 1.0
        node.deg = 0
        node.id = id
        
        return node
    end
end

function push_sum(prob::SeperableQuadratic, x_inits::Dict{Int, Array{Float64, 2}};
                  MAX_CYCLES::Int64=500, recordx::Bool=false)

    # Convenience variables
    graph = prob.graph # communication graph
    n = prob.dim_opt # optimization var dimension
    N = prob.N # number of nodes

    # Initialize Push Sum Node structs and node neighborhoods
    nodes = [PushSumNode(n, x_inits[i], i) for i in 1:N]
    neighs = [neighbors(graph, i) for i in 1:N]

    # Initialize variables to track convergence
    if recordx
        xhist = Dict{Int, Array{Float64, 2}}()
        for i in 1:N
            xhist[i] = zeros(n, MAX_CYCLES)
            xhist[i][:, 1] = x_inits[i]
        end
    end

    # Initialize node degrees
    for i in 1:N
        nodes[i].deg = length(neighs[i])
    end

    # For recording communication overhead
    comm_total = 0
    comm_hist = zeros(MAX_CYCLES, 1)
    comm_unit = sizeof(nodes[1].x) + sizeof(nodes[1].y)
    # so we don't need to call sizeof 25,000 times

    # Main optimization loop
    for t in 2:MAX_CYCLES
        # Alpha update
        alpha = t^-0.5

        # Local node updates
        for node in nodes
            # Communication round
            node.w = zeros(size(node.w))
            node.y = 0.0
            for j in neighs[node.id]  
                node.w += nodes[j].x_prev ./ nodes[j].deg
                node.y += nodes[j].y_prev ./ nodes[j].deg

                comm_total += comm_unit
            end

            # Z-update
            node.z = node.w ./ node.y
            
            # Local gradient computation
            grad = local_gradient(prob, node.id, node.z)

            # X-update
            node.x = node.w - alpha * grad 

            if recordx
                xhist[node.id][:, t] = node.x
            end
        end

        for node in nodes
            if recordx
                xhist[node.id][:, t] = node.x
            end

            node.x_prev = node.x
            node.y_prev = node.y
        end    

        # Record communication cost
        comm_hist[t] = comm_unit
    end

    fvals = Array{Float64}(undef, (n, N))
    for node in nodes
        fvals[:, node.id] = node.x
    end

    return fvals, xhist, comm_hist
end
