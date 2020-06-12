function norm2_error(x::Array{Float64, 2}, xcent::Array{Float64, 2})
    # column 2 is time
    T = size(x, 2)

    error = Array{Float64}(undef, (T, 1))
    for t in 1:T
        error[t] = norm(x[:, t] - xcent)
    end

    return error
end

function norm2_error(xdict::Dict{Int, Array{Float64, 2}},
                     xcent::Array{Float64, 2})
    # column 2 is time
    ks = collect(keys(xdict)) 
    N = length(ks)
    T = size(xdict[1], 2)

    errors = Array{Float64}(undef, (N, T))
    for key in keys(xdict)
        errors[key, :] = norm2_error(xdict[key], xcent)
    end

    return errors
end
