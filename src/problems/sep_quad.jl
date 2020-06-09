mutable struct SeperableQuadratic
    A::Dict{Int64, Array{Float64, 2}}
    b::Dict{Int64, Array{Float64, 2}}
    c::Dict{Int64, Float64}

    dim_opt::Int64
    N::Int64
    graph::SimpleGraph

    function SeperableQuadratic(dim_opt::Int,  graph::SimpleGraph)
        prob = new() 
        prob.A = Dict{Int64, Array{Float64, 2}}()
        prob.b = Dict{Int64, Array{Float64, 2}}()
        prob.c = Dict{Int64, Float64}()

        prob.dim_opt = dim_opt
        prob.N = nv(graph)
        prob.graph = graph

        for i in 1:prob.N
            prob.A[i] = rand_PSD(dim_opt)
            prob.b[i] = rand(dim_opt, 1)
            prob.c[i] = rand()
        end

        return prob
    end
end

function get_max_cond(prob::SeperableQuadratic)
    maxcond = 0.0
    for i = 1:prob.N
        c = cond(prob.A[i])
        if c > maxcond
            maxcond = c
        end
    end
    return maxcond
end

function get_joint_prob(prob::SeperableQuadratic)
    Afull = zeros(prob.dim_opt, prob.dim_opt)
    bfull = zeros(prob.dim_opt, 1)
    cfull = 0.0

    for i in 1:prob.N
        Afull += prob.A[i]
        bfull += prob.b[i]
        cfull += prob.c[i]
    end

    return Afull, bfull, cfull
end

function solve_centralized(prob::SeperableQuadratic)
    A, b, c = get_joint_prob(prob)
    x_cent = (2.0 * A)\(-b)

    return x_cent
end

function solve_local(prob::SeperableQuadratic)
    x_locals = Dict{Int, Array{Float64, 2}}()

    for id in 1:prob.N
        x_locals[id] = (2.0 * prob.A[id]) \ -prob.b[id]
    end

    return x_locals
end

function local_gradient(prob::SeperableQuadratic, id::Int, x::Array{Float64, 2})
    return 2.0 * prob.A[id] * x + prob.b[id]
end
