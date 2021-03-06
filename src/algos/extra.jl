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
               MAX_CYCLES::Int64=500, recordx::Bool=false, alpha::Float64=0.001)

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

    # Setup variables to analyze convergence
    if recordx
        xhist = Dict{Int, Array{Float64, 2}}()
        for i in 1:N
            xhist[i] = zeros(n, MAX_CYCLES)
            xhist[i][:, 1] = x_inits[i]
        end
    end

    # For recording communication overhead
    comm_total = 0
    comm_hist = zeros(MAX_CYCLES, 1)
    comm_unit = sizeof(nodes[1].xk1) # so we don't need to call sizeof 25,000 times

    # Optimization process main loop
    for t in 2:MAX_CYCLES
        # Node by Node inner loop
        for node in nodes
            # Communication with neighbors
            nxk = zeros(n, 1)
            nxk1 = zeros(n, 1)
            for j in neighs[node.id]
                nxk += Wt[node.id, j] * nodes[j].xk_prev
                nxk1 += W[node.id, j] * nodes[j].xk1_prev
                
                # Recording communication cost here, using one communication
                # unit because theoretical we could locally store the
                # neighboring node values from the previous time.
                # We don't in this implementation for readibility.
                comm_total += comm_unit
            end
            
            nxk += Wt[node.id, node.id] * node.xk_prev
            nxk1 += W[node.id, node.id] * node.xk1_prev

            # Compute local gradients
            gradxk = local_gradient(prob, node.id, node.xk_prev)
            gradxk1 = local_gradient(prob, node.id, node.xk1_prev)

            # x-updates
            # TODO fix this x_new business (last time I tried I messed up the
            # algorithm because i wasn't careful).
            if t == 2
                x_new = nxk1 - alpha * gradxk1
            else
                x_new = node.xk1_prev + nxk1 - nxk - alpha * (gradxk1 - gradxk)
            end
            
            node.xk = node.xk1
            node.xk1 = x_new
        end

        # Update node variables for next optimization round
        for node in nodes
            if recordx
                xhist[node.id][:, t] = node.xk1
            end

            node.xk_prev = node.xk
            node.xk1_prev = node.xk1
        end    

        # Record communication cost
        comm_hist[t] = comm_total
    end

    fvals = Array{Float64}(undef, (n, N))
    for node in nodes
        fvals[:, node.id] = node.xk1
    end

    return fvals, xhist, comm_hist
end
