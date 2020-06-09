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
