mutable struct LogisticRegression
    S::Dict{Int, Matrix{Float64}} # Samples for each agent
    y::Dict{Int, Matrix{Float64}} # Classifications

    dim_opt::Int64 # Dimension of optimization problem
    N::Int64 # Number of agents
    graph::SimpleGraph # Communication Graph

    function LogisticRegression(dim_opt::Int, num_samps::Int,
                                graph::SimpleGraph)
        prob = new() 
        prob.S = Dict{Int64, Matrix{Float64}}()
        prob.y = Dict{Int64, Matrix{Float64}}()

        prob.dim_opt = dim_opt
        prob.N = nv(graph)
        prob.graph = graph

        for i in 1:prob.N
            # Generate problem data
        end

        return prob
    end
end

function evaluate(prob::LogisticRegression, x::Matrix{Float64})

end

function get_joint_prob(prob::LogisticRegression)

    return joint
end

function solve_centralized(prob::LogisticRegression)

    return x_cent
end

function solve_local(prob::LogisticRegression)

    return x_locals
end

function local_gradient(prob::LogisticRegression,
                        id::Int, x::Matrix{Float64})

    return 
end
