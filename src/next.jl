mutable struct NEXTNode
    id::Int64
    x::Array{Float64, 2}
    y::Array{Float64, 2}
    z::Array{Float64, 2}
    x_prev::Array{Float64, 2}
    y_prev::Array{Float64, 2}
    pit::Array{Float64, 2}

    J::Array{Float64, 2} # Precomputed inverse for PL-NEXT
    A::Array{Float64, 2} # Precomputed inverse for PL-NEXT
    # This probably isn't the right place for this but temporary

    function NEXTNode(dim::Tuple{Int, Int}, x_init::Array{Float64, 2}, id::Int)
        node = new()
        node.x = x_init
        node.x_prev = x_init
        node.y = zeros(dim)
        node.y_prev = zeros(dim)
        node.z = zeros(dim)
        node.pit = zeros(dim)
        node.id = id
        node.J = zeros(dim[1], dim[1])
        node.A = zeros(dim[1], dim[1])
        
        return node
    end
end

function next(prob::SqDistTargetLoc, x_inits::Dict{Int, Array{Float64, 2}};
              x_cent::Union{Array{Float64, 2}, Nothing}=nothing,
              MAX_CYCLES::Int64=500, recordx::Bool=false,
              verbose::Bool=false)
    # The math for this method is taken from Section IV.A of 
    # NEXT: In-network Nonconvex Optimization by Di Lorenzo et al

    # Convenience variables
    graph = prob.graph # communication graph
    n = prob.dim_opt # optimization var dimension
    N = prob.N # number of nodes
    T = prob.T # number of nodes

    # Problem hyper parameters
    tau = 10.0
    alpha0 = 0.1
    mu = 0.01

    # Generate weighting matrix 
    W = generate_metropolis_weights(graph)

    # Initialize node structs and neighborhoods
    nodes = [NEXTNode((n, T), x_inits[i], i) for i in 1:N]
    neighs = [neighbors(graph, i) for i in 1:N]

    # Initialize convergence analysis variables
    error = zeros(N, MAX_CYCLES - 1)
    xhist = Dict{Int, Array{Float64, 3}}()
    if recordx
        for i in 1:N
            xhist[i] = zeros(n, T, MAX_CYCLES)
            xhist[i][:, :, 1] = x_inits[i]
        end
    end

    # Initialize node y values: y(0) = grad(x(0))
    !verbose ? nothing : println("Initialization: ")
    for node in nodes
        node.y = local_gradient(prob, node.id, x_inits[node.id])
        node.y_prev = node.y
        node.pit = (N - 1) .* node.y 

        w = prob.sp[:, node.id] 

        # Following paper definition
        node.A = 4.0 * w * w' + 2.0 * norm(w)^2 .* 1.0I(n)
        node.J = inv(node.A + tau .* 1.0I(n))
        
        !verbose ? nothing : println()
        !verbose ? nothing : println("node: ", node.id)
        !verbose ? nothing : println("x: ", round.(node.x, digits=3))
        !verbose ? nothing : println("y: ", round.(node.y, digits=3))
        !verbose ? nothing : println("pit: ", round.(node.pit, digits=3))
        !verbose ? nothing : println("J: ", round.(node.J, digits=3))
        !verbose ? nothing : println("A: ", round.(node.A, digits=3))
    end

    alpha = alpha0
    # Main Optimization loop
    !verbose ? nothing : println("Iterations %%%%%%%%%%%%%%%%%%%%")
    for time in 2:MAX_CYCLES
        if time != 2
            alpha = alpha * (1 - mu * alpha)
        end

        # Local SCA Approximations
        !verbose ? nothing : println("SCA *************")
        for node in nodes
            !verbose ? nothing : println("node: ", node.id)

            xtild = pl_convex_approx(prob, node, tau)
            node.z = node.x_prev + alpha .* (xtild - node.x_prev)

            !verbose ? nothing : println("xtild: ", round.(xtild, digits=3))
        end

        # Optimization variable updates with consensus
        #println()
        !verbose ? nothing : println("Consensus *************")
        for node in nodes
            # Communication phase
            nz = zeros(n, T)
            ny = zeros(n, T)
            for j in neighs[node.id]
                nz += W[node.id, j] * nodes[j].z
                ny += W[node.id, j] * nodes[j].y_prev
            end

            # Regular consensus on points
            node.x = W[node.id, node.id] * node.z + nz

            # Computing gradients
            gradn = local_gradient(prob, node.id, node.x_prev)
            gradn1 = local_gradient(prob, node.id, node.x)

            # dynamic consensus on gradients
            node.y = W[node.id, node.id] * node.y_prev + ny + gradn1 - gradn

            # Update estimate of other nodes' gradients
            node.pit = N .* node.y - gradn1

            # Record variables for convergence analysis
            if x_cent == nothing
                error[node.id, time - 1] = norm(reshape(node.x_prev, :) - reshape(node.x, :))
            else
                error[node.id, time - 1] = norm(reshape(node.x_prev, :) - reshape(x_cent, :))
            end

            if recordx
                xhist[node.id][:, :, time] = node.x
            end
            
            !verbose ? nothing : println("-------------------")
            !verbose ? nothing : println("node: ", node.id)
            !verbose ? nothing : println("z: ", round.(node.z, digits=3))
            !verbose ? nothing : println("x: ", round.(node.x, digits=3))
            !verbose ? nothing : println("y: ", round.(node.y, digits=3))
            !verbose ? nothing : println("pit: ", round.(node.pit, digits=3))
        end

        !verbose ? nothing : println("%%%%%%%%%%%%%%")

        # Update node x_prev and y_prev
        for node in nodes
            node.x_prev = node.x
            node.y_prev = node.y
        end    
    end

    return error, xhist
end

function pl_convex_approx(prob::SqDistTargetLoc, node::NEXTNode, tau::Float64;
                          verbose::Bool=false) 
    # This is NOT setup for arbitrary target dimension size!
    i = node.id
    w = prob.sp[:, i]

    xnew = zeros(size(node.x_prev))

    cnst = 4.0 * norm(w)^2 .* w
    Ai = node.A

    for t in 1:prob.T
        # Per-target x-updates
        xn = node.x_prev[:, t]
        bn = cnst - 4.0*(norm(xn)^2-prob.meas[i, t])*(xn - w) + 8.0.*(w'*xn)*xn

        xh = node.J * (bn - node.pit[:, t] + tau .* xn)
        !verbose ? nothing : println("xh", t, ": ", round.(xh, digits=3))
        !verbose ? nothing : println("bn", t, ": ", round.(bn, digits=3))

        xt = zeros(prob.dim_opt, 1)

        if xh[1] >= 0.0 && xh[1] <= 1.0 && xh[2] < 0.0
            !verbose ? nothing : println("Cond 1")
            xt[1] = bn[1] / (2.0 * Ai[1, 1]) #1-D solution
            xt[2] = 0.0
        elseif xh[1] >= 0.0 && xh[1] <= 1.0 && xh[2] > 1.0
            !verbose ? nothing : println("Cond 2")
            xt[1] = (bn[1] - Ai[1, 2] - Ai[2, 1]) / (2.0 * Ai[1, 1])
            xt[2] = 1.0
        elseif xh[2] >= 0.0 && xh[2] <= 1.0 && xh[1] < 0.0
            !verbose ? nothing : println("Cond 3")
            xt[1] = 0.0
            xt[2] = bn[2] / (2.0 * Ai[2, 2])
        elseif xh[2] >= 0.0 && xh[2] <= 1.0 && xh[1] > 1.0
            !verbose ? nothing : println("Cond 4")
            xt[1] = 1.0
            xt[2] = (bn[2] - Ai[1, 2] - Ai[2, 1]) / (2.0 * Ai[2, 2])
        else
            !verbose ? nothing : println("Cond 5")
            xt[1] = min(max(xh[1], 0.0), 1.0)
            xt[2] = min(max(xh[2], 0.0), 1.0)
        end

        xnew[:, t] = xt
    end 

    return xnew
end

