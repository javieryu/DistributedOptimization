mutable struct ExtraNode
    id::Int64
    xk::Array{Float64, 2}
    xk1::Array{Float64, 2}
    xk_prev::Array{Float64, 2}
    xk1_prev::Array{Float64, 2}

    function ExtraNode(dim::Int, x_init::Array{Float64, 2}, id::Int)
        node = new()
        node.xk = zeros(dim, 1)
        node.xk1 = x_init
        node.xk_prev = zeros(dim, 1)
        node.xk1_prev = x_init
        node.id = id
        
        return node
    end
end

function extra(prob::SeperableQuadratic, x_inits::Dict{Int, Array{Float64, 2}};
               MAX_CYCLES::Int64=500, 
               x_cent::Union{Array{Float64, 2}, Nothing}=nothing,
               recordx::Bool=false, alpha::Float64=0.001)

    # Convenience variables
    graph = prob.graph # communication graph
    n = prob.dim_opt # optimization var dimension
    N = prob.N # number of nodes

    # Construct Weighting Matrix (not distributed)
    W = generate_metropolis_weights(graph)
    Wt = (1.0I(N) + W) ./ 2

    # Initialize nodes
    nodes = [ExtraNode(n, x_inits[i], i) for i in 1:N]
    neighs = [neighbors(graph, i) for i in 1:N]

    # Setup variables to asses convergence
    error = zeros(N, MAX_CYCLES - 1)
    xhist = Dict{Int, Array{Float64, 2}}()
    if recordx
        for i in 1:N
            xhist[i] = zeros(n, MAX_CYCLES)
            xhist[i][:, 1] = x_inits[i]
        end
    end

    # Optimization process main loop
    for t in 2:MAX_CYCLES
        # Node by Node inner loop
        for node in nodes
            # Communication with neighbors
            nxk = zeros(n, 1)
            nxk1 = zeros(n, 1)
            for j in vcat(neighs[node.id], node.id)
                nxk += Wt[node.id, j] * nodes[j].xk_prev
                nxk1 += W[node.id, j] * nodes[j].xk1_prev
            end

            # Compute local gradients
            gradxk = local_gradient(prob, node.id, node.xk_prev)
            gradxk1 = local_gradient(prob, node.id, node.xk1_prev)

            # x-updates
            x_new = zeros(n, 1)
            if t == 2
                x_new = nxk1 - alpha * gradxk1
            else
                x_new = node.xk1_prev + nxk1 - nxk - alpha * (gradxk1 - gradxk)
            end
            
            # Update and store relevant values
            if x_cent == nothing
                error[node.id, t - 1] = norm(node.xk1_prev - x_new)
            else
                error[node.id, t - 1] = norm(node.xk1_prev - x_cent)
            end

            if recordx
                xhist[node.id][:, t] = x_new
            end

            node.xk = node.xk1
            node.xk1 = x_new
        end

        # Update node variables for next optimization round
        for node in nodes
            node.xk_prev = node.xk
            node.xk1_prev = node.xk1
        end    
    end

    return error, xhist
end
