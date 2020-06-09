mutable struct DIGNode
    id::Int64
    x::Array{Float64, 2}
    y::Array{Float64, 2}
    x_prev::Array{Float64, 2}
    y_prev::Array{Float64, 2}

    function DIGNode(dim::Int, x_init::Array{Float64, 2}, id::Int)
        node = new()
        node.x = x_init
        node.x_prev = x_init
        node.y = zeros(dim, 1)
        node.y_prev = zeros(dim, 1)
        node.id = id
        
        return node
    end
end

function dig(prob::SeperableQuadratic, x_inits::Dict{Int, Array{Float64, 2}};
                 MAX_CYCLES::Int64=500, 
                 x_cent::Union{Array{Float64, 2}, Nothing}=nothing,
                 recordx::Bool=false, alpha::Float64=0.001)

    # Convenience variables
    graph = prob.graph # communication graph
    n = prob.dim_opt # optimization var dimension
    N = prob.N # number of nodes

    # Generate weighting matrix 
    # THIS STEP IS NOT DISTRIBUTED
    W = generate_metropolis_weights(graph)

    # Initialize node structs and neighborhoods
    nodes = [DIGNode(n, x_inits[i], i) for i in 1:N]
    neighs = [neighbors(graph, i) for i in 1:N]

    # Initialize convergence analysis variables
    error = zeros(N, MAX_CYCLES - 1)
    xhist = Dict{Int, Array{Float64, 2}}()
    if recordx
        for i in 1:N
            xhist[i] = zeros(n, MAX_CYCLES)
            xhist[i][:, 1] = x_inits[i]
        end
    end

    # Initialize node y values: y(0) = grad(x(0))
    for i in 1:N
        nodes[i].y = local_gradient(prob, i, x_inits[i])
        nodes[i].y_prev = nodes[i].y
    end

    # Main Optimization loop
    for t in 2:MAX_CYCLES

        # Local node updates
        for node in nodes
            # Communication round
            nx = zeros(n, 1)
            ny = zeros(n, 1)
            for j in vcat(neighs[node.id], node.id)
                nx += W[node.id, j] * nodes[j].x_prev
                ny += W[node.id, j] * nodes[j].y_prev
            end

            # x-update
            x_new = nx - alpha * node.y_prev

            # Compute gradients
            gradxk = local_gradient(prob, node.id, node.x_prev)
            gradxk1 = local_gradient(prob, node.id, x_new)

            # y-update
            y_new = ny + gradxk1 - gradxk

            # Record variables for convergence analysis
            if x_cent == nothing
                error[node.id, t - 1] = norm(node.x_prev - x_new)
            else
                error[node.id, t - 1] = norm(node.x_prev - x_cent)
            end

            if recordx
                xhist[node.id][:, t] = x_new
            end

            # Update node x and y values
            node.x = x_new
            node.y = y_new
        end

        # Update node x_prev and y_prev
        for node in nodes
            node.x_prev = node.x
            node.y_prev = node.y
        end    
    end

    return error, xhist
end
