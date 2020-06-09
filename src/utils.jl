function generate_random_graph(NV::Int, NE::Int)
    # Generates a random connected graph
    # nv : number of vertices in graph
    # ne : number of edges in graph
    #
    CONNECT_LIMIT = 50

    graph = SimpleGraph(NV, NE)

    for i in 1:CONNECT_LIMIT
        if is_connected(graph)
            break
        else
            graph = SimpleGraph(NV, NE)
        end
    end

    if !is_connected(graph)
        println("generate_random_graph: WARNING graph is not connected.")
    end

    return graph
end

function rand_PSD(dim::Int64)
    R = rand(dim, dim)
    A = 0.5 .* (R' + R)
    A = A + dim .* 1.0I(dim)
    
    return A
end

function generate_random_delaunay_graph(N::Int64)
    graph = SimpleGraph(N)

    tess = DelaunayTessellation(N)
    width = VoronoiDelaunay.max_coord - VoronoiDelaunay.min_coord
    a = Point2D[Point(VoronoiDelaunay.min_coord + rand() * width,
                      VoronoiDelaunay.min_coord + rand() * width) for i in 1:N]
    push!(tess, a)

#    plt = plot()
#    x, y = getplotxy(voronoiedges(tess))
#    plot!(plt, x, y, c=:red, lab="voronoi edges")
#    x, y = getplotxy(delaunayedges(tess))
#    plot!(plt, x, y, c=:blue, lab="delaunayedges")
#    display(plt)

    point_lookup = Dict(a[i] => i for i in 1:N)

    for edge in delaunayedges(tess)
        i = point_lookup[geta(edge)]
        j = point_lookup[getb(edge)]
        add_edge!(graph, i, j)
    end

    return graph, tess
end

function generate_disk_graph(N::Int)
    r = 0.35
    for i = 1:1000
        g, dists, pts = euclidean_graph(N, 2, cutoff=r)

        if is_connected(g)
            return (g, pts)
        end
    end
    
    println("Graph gen failed to produce a connected disk graph.")
    return 0
end

function generate_metropolis_weights(graph::SimpleGraph; eps=1.0)
    N = nv(graph)
    W = zeros(N, N)

    L = laplacian_matrix(graph)
    degs = [L[i, i] for i in 1:N]

    for i in 1:N
        for j in 1:N
            if has_edge(graph, i, j) && i != j
                W[i, j] = 1 / (max(degs[i], degs[j]) + eps)
            end
        end
    end

    for i in 1:N
        W[i, i] = 1.0 - sum(W[i, :])
    end

    return W
end
