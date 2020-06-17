# An assortment of functions for various plotting activites

function plot_all_errors!(plt, error::Array{Float64, 2})
    T = size(error, 2)
    plot!(plt, 1:T, error')
end

function plot_bounded_errors!(plt, errors::Array{Float64, 2}, color,
                            label::String; until=nothing)
     
    until == nothing ? until = size(errors, 2) : nothing
    T = until # This is a not very readable fix.

    ub = zeros(T, 1)
    lb = zeros(T, 1)
    m = zeros(T, 1)

    for t in 1:T
        ub[t] = maximum(errors[:, t])
        lb[t] = minimum(errors[:, t])
        m[t] = mean(errors[:, t])
    end

    plot!(plt, m, fillrange=lb, fillalpha=0.3, c=color, lab="")
    plot!(plt, m, fillrange=ub, fillalpha=0.3, c=color, lab=label)
end

function plot_stat!(plt, stat::Array{Float64, 2},
                    color::Symbol, label::String)
    plot!(plt, stat, c=color, lab="label") 
end

function plot_mean_error!(plt, errors::Array{Float64, 2},
                            color::Symbol, label::String)
    T = size(errors, 2)
    m = zeros(T, 1)

    for t in 1:T
        m[t] = mean(errors[:, t])
    end

    plot!(plt, m, c=color, lab=label)
end

function plot_xhist!(plt, xhist::Dict{Int, Array{Float64, 2}}, color;
                     until=nothing)
    if size(xhist[1], 1) != 2
        println("Dimension of x is not 2")
        return
    end

    until == nothing ? until = size(xhist[1], 2) : until

    for key in keys(xhist)
        plot!(plt, xhist[key][1, 1:until], xhist[key][2, 1:until], c=color)
        scatter!(plt, [xhist[key][1, until]], [xhist[key][2, until]], c=color)
    end
end

function plot_sq_dis_traj!(plt, prob::SqDistTargetLoc,
                           xhist::Dict{Int64, Array{Float64, 3}})
    # xhist: (1:2, 1:T, 1:MAX_CYCLES)    
    T = prob.T
    N = prob.N

    pcols = distinguishable_colors(N+1, RGB(1,1,1))

    for t in 1:T
        for n in 1:N
            plot!(plt, xhist[n][1, t, :], xhist[n][2, t, :], c=pcols[n+1])
        end

        gt = prob.grnd_tru[:,t]
        scatter!(plt, [gt[1]], [gt[2]], c=:black)
    end

    for n in 1:N
        sp = prob.sp[:, n]
        #scatter!(plt, [sp[1]], [sp[2]], c=pcols[n+1])
    end
end

function plot_xcent!(plt, x_cent::Array{Float64, 2}, color)
    if size(x_cent, 1) != 2
        println("Dimension of x_cent is not 2, cannot display")
        return
    end

    scatter!(plt, [x_cent[1]], [x_cent[2]], c=color)
end

function plot_xinits!(plt, x_inits::Dict{Int, Array{Float64, 2}}, color)
    if size(x_inits[1], 1) != 2
        println("Dimension of x_init is not 2, cannot display")
        return
    end

    data = zeros(2, length(collect(keys(x_inits))))

    for key in keys(x_inits)
        data[:, key] = x_inits[key]
    end

    scatter!(plt, data[1, :], data[2, :], c=color)
end

function get_xhist_limits(xhist::Dict{Int, Matrix{Float64}})
    # Returns the bounds on dimension for each of the dimensions
    # across all of the decision var tajectories
    
    n = size(xhist[1], 1)
    limits = zeros(n, 2) # [min, max] in each dimension

    for key in keys(xhist)
        for i in 1:n
            ub = maximum(xhist[key][i, :])
            lb = minimum(xhist[key][i, :])

            if ub > limits[i, 2]
                limits[i, 2] = ub
            end

            if lb < limits[i, 1]
                limits[i, 1] = lb
            end
        end
    end

    return limits
end

function plot_cost_contour!(plt, prob::SeperableQuadratic,
                         limits::Matrix{Float64}, resolution::Int)
    if prob.dim_opt != 2
        println("Cannot generate 2d cost map of a not 2d input.")
        return nothing
    end

    A, b, c = get_joint_prob(prob)

    x1_range = range(limits[1,1], limits[1,2], length=resolution)
    x2_range = range(limits[2,1], limits[2,2], length=resolution)

    cost(x1, x2) = begin 
        [x1; x2]' * A * [x1; x2] + dot(b, [x1; x2])
    end

    contour!(plt, x1_range, x2_range, cost, c=:grays)
end

