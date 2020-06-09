mutable struct DDANode
    id::Int64
    deg::Int64
    x::Array{Float64, 2}
    x_prev::Array{Float64, 2}
    z::Array{Float64, 2}
    z_prev::Array{Float64, 2}

    function DDANode(dim::Int, x_init::Array{Float64, 2}, id::Int)
        node = new()
        node.x = x_init
        node.x_prev = x_init
        node.z = zeros(dim, 1)
        node.z_prev = zeros(dim, 1)
        node.deg = 0
        node.id = id
        
        return node
    end
end

function dda(prob::SeperableQuadratic, x_inits::Dict{Int, Array{Float64, 2}};
             MAX_CYCLES::Int64=500, 
             x_cent::Union{Array{Float64, 2}, Nothing}=nothing,
             recordx::Bool=false)

    # Convenience variables
    graph = prob.graph # communication graph
    n = prob.dim_opt # optimization var dimension
    N = prob.N # number of nodes

    # Initialize DDA node structs and node neighborhoods
    nodes = [DDANode(n, x_inits[i], i) for i in 1:N]
    neighs = [neighbors(graph, i) for i in 1:N]

    # Initialize variables for convergence analysis
    error = zeros(N, MAX_CYCLES - 1)
    xhist = Dict{Int, Array{Float64, 2}}()

    if recordx
        for i in 1:N
            xhist[i] = zeros(n, MAX_CYCLES)
            xhist[i][:, 1] = x_inits[i]
        end
    end

    # Initialize local node degrees
    for i in 1:N
        nodes[i].deg = length(neighs[i])
    end

    # Compute weighting matrix
    # THIS STEP IS NOT DISTRIBUTED
    # SO THE METHOD IS NOT COMPLETELY DISTRIBUTED
    P = generate_metropolis_weights(graph)

    # Main optimization loop
    for t in 2:MAX_CYCLES
        # alpha update 
        alpha = t^-0.5

        # Local node updates
        for node in nodes
            # Communication Rounds
            node.z = zeros(n, 1)
            for j in vcat(neighs[node.id], node.id)
                node.z += P[node.id, j] .* nodes[j].z_prev
            end

            # Local gradient updates
            grad = local_gradient(prob, node.id, node.x_prev)

            # Z-update
            node.z += -grad

            # X-update based on proximal function: 1/2 * ||x||^2
            x_new = alpha .* node.z

            # Update convergence analysis variables
            if x_cent == nothing
                error[node.id, t - 1] = norm(node.x - x_new)
            else
                error[node.id, t - 1] = norm(node.x - x_cent)
            end

            if recordx
                xhist[node.id][:, t] = x_new
            end

            # node x value update
            node.x = x_new
        end

        # node x_prev value update
        for node in nodes
            node.x_prev = node.x
            node.z_prev = node.z
        end    
    end

    return error, xhist
end
