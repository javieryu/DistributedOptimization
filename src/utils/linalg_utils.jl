function rand_PSD(dim::Int64)
    R = rand(dim, dim)
    A = 0.5 .* (R' + R)
    A = A + dim .* 1.0I(dim)
    
    return A
end


